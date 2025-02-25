# this is very heavily borrowed from discord user @Mojonero, who kindly shared his s2 starter here: https://discord.com/channels/1079907749569237093/1204133327083147264/1204133327083147264

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from pytorch3dunet.augment.transforms import (
    Compose, LabelToAffinities, Standardize,
    RandomFlip, RandomRotate90
)
from tqdm import tqdm
import multiprocessing
import json
from skimage.morphology import dilation, ball
from multiprocessing import Pool

def ct_normalize(volume, clip_min, clip_max, global_mean, global_std):
    """
    Mimics nnU-Net CT normalization:
      1) Clip to [clip_min, clip_max],
      2) Subtract mean,
      3) Divide by std.
    """
    volume = volume.astype(np.float32)
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - global_mean) / global_std
    return volume

class TargetedTransform:
    """Wrapper for transforms that only applies them to specified data types"""
    def __init__(self, transform, targets):
        self.transform = transform
        self.targets = targets
        self.is_normals = getattr(transform, 'is_normals', False)

    def __call__(self, data_dict):
        result = {}
        for key, value in data_dict.items():
            if key in self.targets:
                # If transform is geometry-based and target is normals
                if key == 'normals' and isinstance(self.transform, (RandomFlip, RandomRotate90)):
                    # Make a normals-specific transform instance
                    normals_transform = type(self.transform)(
                        random_state=self.transform.random_state,
                        is_normals=True,
                        **{
                            k: v for k, v in self.transform.__dict__.items()
                            if k not in ['random_state', 'is_normals']
                        }
                    )
                    result[key] = normals_transform(value)
                else:
                    result[key] = self.transform(value)
            else:
                result[key] = value
        return result

def _chunker(seq, chunk_size):
    """Yield successive 'chunk_size'-sized chunks from 'seq'."""
    for pos in range(0, len(seq), chunk_size):
        yield seq[pos:pos + chunk_size]

def compute_bounding_box_3d(mask):
    """
    Given a 3D boolean array (True where labeled, False otherwise),
    returns (minz, maxz, miny, maxy, minx, maxx).
    If there are no nonzero elements, returns None.
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    minz, miny, minx = nonzero_coords.min(axis=0)
    maxz, maxy, maxx = nonzero_coords.max(axis=0)
    return (minz, maxz, miny, maxy, minx, maxx)


def bounding_box_volume(bbox):
    """
    Given a bounding box (minz, maxz, miny, maxy, minx, maxx),
    returns the volume (number of voxels) inside the box.
    """
    minz, maxz, miny, maxy, minx, maxx = bbox
    return ((maxz - minz + 1) *
            (maxy - miny + 1) *
            (maxx - minx + 1))


def _check_patch_chunk(chunk, sheet_label, patch_size, bbox_threshold=0.5, label_threshold=0.05):
    """
    Worker function to check each patch in 'chunk' with both:
      - bounding box coverage >= bbox_threshold
      - overall labeled voxel ratio >= label_threshold
    """
    pD, pH, pW = patch_size
    valid_positions = []

    for (z, y, x) in chunk:
        patch = sheet_label[z:z + pD, y:y + pH, x:x + pW]
        # Compute bounding box of nonzero pixels in this patch
        bbox = compute_bounding_box_3d(patch > 0)
        if bbox is None:
            # No nonzero voxels at all -> skip
            continue

        # 1) Check bounding box coverage
        bb_vol = bounding_box_volume(bbox)
        patch_vol = patch.size  # pD * pH * pW
        if bb_vol / patch_vol < bbox_threshold:
            continue

        # 2) Check overall labeled fraction
        labeled_ratio = np.count_nonzero(patch) / patch_vol
        if labeled_ratio < label_threshold:
            continue

        # If we passed both checks, add to valid positions
        valid_positions.append((z, y, x))

    return valid_positions


def find_label_bounding_box(sheet_label_array,
                            chunk_shape=(192, 192, 192)) -> tuple:
    """
    Find the minimal bounding box (minz, maxz, miny, maxy, minx, maxx)
    that contains all non-zero voxels in `sheet_label_array`.

    :param sheet_label_array: a 3D zarr array of shape (D, H, W)
    :param chunk_shape: (chunk_z, chunk_y, chunk_x) to read from disk at once
    :return: (minz, maxz, miny, maxy, minx, maxx)
             such that sheet_label_array[minz:maxz+1, miny:maxy+1, minx:maxx+1]
             contains all non-zero voxels.
             If no non-zero voxel is found, returns (0, -1, 0, -1, 0, -1)
    """
    D, H, W = sheet_label_array.shape

    # Initialize bounding box to "empty"
    minz, miny, minx = D, H, W
    maxz = maxy = maxx = -1

    # We'll track total chunks for a TQDM progress bar
    # so we know how many chunk-reads we're doing
    num_chunks_z = (D + chunk_shape[0] - 1) // chunk_shape[0]
    num_chunks_y = (H + chunk_shape[1] - 1) // chunk_shape[1]
    num_chunks_x = (W + chunk_shape[2] - 1) // chunk_shape[2]
    total_chunks = num_chunks_z * num_chunks_y * num_chunks_x

    with tqdm(desc="Finding label bounding box", total=total_chunks) as pbar:
        for z_start in range(0, D, chunk_shape[0]):
            z_end = min(D, z_start + chunk_shape[0])
            for y_start in range(0, H, chunk_shape[1]):
                y_end = min(H, y_start + chunk_shape[1])
                for x_start in range(0, W, chunk_shape[2]):
                    x_end = min(W, x_start + chunk_shape[2])
                    # Read just this chunk from the zarr
                    chunk = sheet_label_array[z_start:z_end, y_start:y_end, x_start:x_end]

                    if chunk.any():  # means there's at least one non-zero voxel
                        # Find the local coords of non-zero voxels
                        nz_idx = np.argwhere(chunk > 0)  # shape (N, 3)
                        # Shift them by the chunk offset
                        nz_idx[:, 0] += z_start
                        nz_idx[:, 1] += y_start
                        nz_idx[:, 2] += x_start

                        cminz = nz_idx[:, 0].min()
                        cmaxz = nz_idx[:, 0].max()
                        cminy = nz_idx[:, 1].min()
                        cmaxy = nz_idx[:, 1].max()
                        cminx = nz_idx[:, 2].min()
                        cmaxx = nz_idx[:, 2].max()

                        # Update global bounding box
                        minz = min(minz, cminz)
                        maxz = max(maxz, cmaxz)
                        miny = min(miny, cminy)
                        maxy = max(maxy, cmaxy)
                        minx = min(minx, cminx)
                        maxx = max(maxx, cmaxx)

                    pbar.update(1)

    # If maxz remains -1, that means no non-zero voxel was found at all
    return (minz, maxz, miny, maxy, minx, maxx)

class ZarrSegmentationDataset3D(Dataset):
    def __init__(self,
                 volume_path: Path,
                 sheet_label_path: Path,
                 normals_path: Path,
                 patch_size=(128, 128, 128),
                 min_labeled_ratio=0.8,
                 xy_offsets=None,
                 z_offsets=None,
                 transforms_list=None,
                 cache_file: Path = None,
                 use_cache: bool = True
                 ):

        self.volume_path = volume_path
        self.sheet_label_path = sheet_label_path
        self.normals_path = normals_path
        self.patch_size = patch_size
        self.min_labeled_ratio = min_labeled_ratio

        self.xy_offsets = xy_offsets
        self.z_offsets = z_offsets
        self.use_affinities = (xy_offsets is not None) and (z_offsets is not None)


        self.cache_file = cache_file
        self.use_cache = use_cache



        # open up our zarr arrays
        print(f"Opening arrays from {volume_path} and {sheet_label_path} and {normals_path}...")
        self.volume_array = zarr.open(str(volume_path), mode='r')
        print(f"volume_array shape: {self.volume_array.shape}")
        self.sheet_label_array = zarr.open(str(sheet_label_path), mode='r')
        print(f"sheet_label_array shape: {self.sheet_label_array.shape}")
        self.normals_array = zarr.open(str(normals_path), mode='r')
        print(f"normals_array shape: {self.normals_array.shape}")

        # set random state so we can reproduce for reasons
        self.random_state = np.random.RandomState(47)

        # set zscore as our standardization scheme, same to all chans
        self.standardizer = Standardize(channelwise=False)

        # If we're using affinities, create the transform; otherwise set to None
        if self.use_affinities:
            self.affinity_transform = LabelToAffinities(
                offsets=xy_offsets,
                z_offsets=z_offsets,
                append_label=False,
                aggregate_affinities=False
            )
            print("AffinityTransform parameters:")
            print(f"  xy_offsets = {xy_offsets}")
            print(f"  z_offsets = {z_offsets}")
        else:
            self.affinity_transform = None


        # gather transforms with their targets and probabilities
        if transforms_list:
            targeted_transforms = []
            print("Transforms list details:")
            for transform, targets in transforms_list:
                print(f"- Transform class: {type(transform).__name__}")
                if hasattr(transform, '__dict__'):
                    print("  Transform parameters:", vars(transform))
                else:
                    print("  Transform parameters are not exposed via __dict__")
                print(f"  Targets: {targets}")
                targeted_transforms.append(TargetedTransform(transform, targets))

            self.transforms = Compose(targeted_transforms)
        else:
            self.transforms = None

        print(f"using transforms: {self.transforms}")

        # check if we have single-volume or multi-volume
        # e.g., single-volume should be (Z, Y, X) or (Z, Y, X, C)
        # multi-volume mask should be (N, Z, Y, X) or (N, Z, Y, X, C)
        self.ndim = self.sheet_label_array.ndim

        if self.ndim == 3 or self.ndim == 4:
            # single volume
            self._is_single_volume = True
            # shape is either (Z, Y, X) or (Z, Y, X, C)
            # We interpret "volume_shape" as the full shape minus any channels
            # but for mask, presumably it's just (Z, Y, X).
            self._volume_depth = self.sheet_label_array.shape[0]
            self._volume_height = self.sheet_label_array.shape[1]
            self._volume_width = self.sheet_label_array.shape[2]
            # We'll store only one 'volume_idx' = 0
            self.n_volumes = 1
        else:
            # Multi-volume scenario
            self._is_single_volume = False
            # shape is (N, Z, Y, X) or (N, Z, Y, X, C)
            self.n_volumes = self.sheet_label_array.shape[0]
            self._volume_depth = self.sheet_label_array.shape[1]
            self._volume_height = self.sheet_label_array.shape[2]
            self._volume_width = self.sheet_label_array.shape[3]


        # if cache file exists, load valid patches.
        # otherwise, compute them and optionally write to cache.
        self.valid_patches = []
        if self.use_cache and self.cache_file is not None and self.cache_file.exists():
            print(f"Cache file found at {self.cache_file}. Loading valid patches...")
            with open(self.cache_file, 'r') as f:
                # Note: these will load back as lists, so 'start_pos' will be a list, etc.
                self.valid_patches = json.load(f)
            print(f"Loaded {len(self.valid_patches)} valid patches from cache.")
        else:
            # compute valid patches
            self._find_valid_patches()
            print(f"Found {len(self.valid_patches)} valid patches.")

            # write to cache if use_cache
            if self.use_cache and self.cache_file is not None:
                print(f"Saving valid patches to {self.cache_file}...")
                with open(self.cache_file, 'w') as f:
                    json.dump(self.valid_patches, f)

    def _find_valid_patches(self,
                            bbox_threshold=0.97,  # bounding-box coverage fraction
                            label_threshold=0.10,  # minimum % of voxels labeled
                            num_workers=16):
        """
        Finds patches that contain:
          - a bounding box of labeled voxels >= bbox_threshold fraction of the patch volume
          - an overall labeled voxel fraction >= label_threshold
        """
        # Decide which volume to use as reference
        if self._is_single_volume:
            sheet_label = self.sheet_label_array
        else:
            sheet_label = self.sheet_label_array[0]

        pD, pH, pW = self.patch_size

        # 1. bounding box (outer) on the reference label array
        minz, maxz, miny, maxy, minx, maxx = find_label_bounding_box(sheet_label)

        # 2. generate possible start positions
        z_step = pD // 2
        y_step = pH // 2
        x_step = pW // 2
        all_positions = []
        for z in range(minz, maxz - pD + 2, z_step):
            for y in range(miny, maxy - pH + 2, y_step):
                for x in range(minx, maxx - pW + 2, x_step):
                    all_positions.append((z, y, x))

        # 3. parallel checking
        chunk_size = max(1, len(all_positions) // (num_workers * 2))
        position_chunks = list(_chunker(all_positions, chunk_size))

        print(
            f"Finding valid patches of size: {self.patch_size} "
            f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
        )

        valid_positions_ref = []
        with Pool(processes=num_workers) as pool:
            results = [
                pool.apply_async(
                    _check_patch_chunk,
                    (
                        chunk,
                        sheet_label,
                        self.patch_size,
                        bbox_threshold,  # pass bounding box threshold
                        label_threshold  # pass label fraction threshold
                    )
                )
                for chunk in position_chunks
            ]
            for r in tqdm(results, desc="Checking patches", total=len(results)):
                valid_positions_ref.extend(r.get())

        # 4. replicate if multi-volume
        valid_patches = []
        if self._is_single_volume:
            for (z, y, x) in valid_positions_ref:
                valid_patches.append({'volume_idx': 0, 'start_pos': [z, y, x]})
        else:
            for volume_idx in range(self.n_volumes):
                for (z, y, x) in valid_positions_ref:
                    valid_patches.append({'volume_idx': volume_idx, 'start_pos': [z, y, x]})

        self.valid_patches = valid_patches
        print(
            f"Found {len(valid_positions_ref)} valid patches in reference volume. "
            f"Total {len(self.valid_patches)} across all volumes."
        )

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_info = self.valid_patches[idx]
        volume_idx = patch_info['volume_idx']
        z0, y0, x0 = patch_info['start_pos']
        dz, dy, dx = self.patch_size

        patch_slice = np.s_[z0:z0+dz, y0:y0+dy, x0:x0+dx] # get the corner indices, expand by patch size

        # --- Volume data ---
        if self._is_single_volume:
            volume_data = self.volume_array[patch_slice]
        else:
            volume_data = self.volume_array[volume_idx][patch_slice]

        images = volume_data.astype(np.float32)
        images /= 255.0
        images = ct_normalize(
            images,
            clip_min=43.0,
            clip_max=240.0,
            global_mean=129.9852752685547,
            global_std=44.020145416259766
        )

        # --- sheet data  ---
        if self._is_single_volume:
            sheet_label = self.sheet_label_array[patch_slice].astype(np.float32)
        else:
            sheet_label = self.sheet_label_array[volume_idx][patch_slice].astype(np.float32)

        # normalize to 0, 1
        sheet_label /= 255.0
        sheet_label = (sheet_label > 0).astype(bool)
        kern = ball(radius=1)  # applying a very tiny dilation to my labels as the model is having a hard time
        sheet_label = dilation(sheet_label, kern)
        sheet_label=sheet_label.astype(np.float32)

        # --- Normals data ---
        if self._is_single_volume:
            normals = self.normals_array[patch_slice].astype(np.float32)
        else:
            normals = self.normals_array[volume_idx][patch_slice].astype(np.float32)

        # shape is (Z, Y, X, 3)
        # scale normals from uint16 back to float
        normals /= 32767.5  # now in [0, 2]
        normals -= 1.0  # now in [-1, 1]

        # TODO: fix normal writing script so we dont have to transpose like this
        # transpose the normals to channel C, Z, Y, X for compatibility with torch.
        normals = normals.transpose(3, 0, 1, 2).copy() # (Z, Y, X, 3) => (3, Z, Y, X)

        # i normalized them when i wrote them, this is just here as a placeholder
        # normalize the....normals....
        # magnitudes = np.linalg.norm(normals, axis=-1)  # shape (Z, Y, X)
        # valid = (magnitudes > 0)
        # normals[valid] /= magnitudes[valid, np.newaxis]

        # --- Affinity maps (optional) ---
        if self.use_affinities and self.affinity_transform is not None:
            affinity_maps = self.affinity_transform(sheet_label)
        else:
            affinity_maps = None

        # add a dimension just so i can stop trying to deal with 3d v 4d in transforms
        images = images[np.newaxis, ...]  # (Z, Y, X) -> (1, Z, Y, X)
        sheet_label = sheet_label[np.newaxis, ...]  # (Z, Y, X) -> (1, Z, Y, X)

        # --- augs --- #
        if self.transforms:
            data_dict = {
                'image': images,
                'sheet_label': sheet_label,
                'normals': normals
            }
            if self.use_affinities and affinity_maps is not None:
                data_dict['affinity'] = affinity_maps

            data_dict = self.transforms(data_dict)

            images = data_dict['image']
            sheet_label = data_dict['sheet_label']
            normals = data_dict['normals']
            if self.use_affinities and 'affinity' in data_dict:
                affinity_maps = data_dict['affinity']

        # --- Convert to torch tensors ---
        images = torch.from_numpy(np.ascontiguousarray(images))
        sheet_label = torch.from_numpy(np.ascontiguousarray(sheet_label))
        normals = torch.from_numpy(np.ascontiguousarray(normals))

        if self.use_affinities and affinity_maps is not None:
            affinity_maps = torch.from_numpy(np.ascontiguousarray(affinity_maps))
        else:
            affinity_maps = None

        if self.use_affinities:
            return images, sheet_label, normals, affinity_maps
        else:
            return images, sheet_label, normals
        # now doing this before augs
        # images = images.unsqueeze(0)  # (Z, Y, X) => (1, Z, Y, X) or (1, Z, Y, X, C)
        # sheet_label = sheet_label.unsqueeze(0)  # (Z, Y, X) => (1, Z, Y, X)



    def close(self):
        """Close the Zarr arrays."""
        self.volume_array.store.close()
        self.sheet_label_array.store.close()
        self.normals_array.store.close()
