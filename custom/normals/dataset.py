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

def _check_patch_chunk(chunk, sheet_label, patch_size, min_labeled_ratio):
    """
    Worker function: given a list of (z, y, x) and shared parameters,
    return only those positions that meet the min_labeled_ratio criterion.
    """
    pD, pH, pW = patch_size
    valid_positions = []
    for (z, y, x) in chunk:
        patch = sheet_label[z:z + pD, y:y + pH, x:x + pW]
        # Check ratio of >0 vs. total
        if np.sum(patch > 0) / patch.size >= min_labeled_ratio:
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

def _estimate_single_patch_memory(self):
    """
    Estimate (in MB) the memory usage for a single patch
    (image + label + normals + affinity maps) in float32.

    This is a naive calculation that assumes:
      - image is (1, Z, Y, X)
      - label is (1, Z, Y, X)
      - normals is (3, Z, Y, X)
      - affinity is (n_offsets, Z, Y, X)
      - float32 => 4 bytes per element
    """
    bytes_per_float = 4  # float32
    Z, Y, X = self.patch_size

    # --- Image: shape = (1, Z, Y, X)
    image_size = 1 * Z * Y * X

    # --- Label: shape = (1, Z, Y, X)
    label_size = 1 * Z * Y * X

    # --- Normals: shape = (3, Z, Y, X)
    normals_size = 3 * Z * Y * X

    # --- Affinity: shape = (#_offsets, Z, Y, X)
    # For example, if we have xy_offsets=[1,3,6] => 3 offsets in xy
    # and z_offsets=[1,3,6] => 3 offsets in z,
    # total offsets = 6 => shape = (6, Z, Y, X)
    num_offsets = len(self.xy_offsets) + len(self.z_offsets)
    affinity_size = num_offsets * Z * Y * X

    total_elements = image_size + label_size + normals_size + affinity_size
    total_bytes = total_elements * bytes_per_float
    total_mb = total_bytes / (1024**2)

    return total_mb

class ZarrSegmentationDataset3D(Dataset):
    def __init__(self,
                 volume_path: Path,
                 sheet_label_path: Path,
                 normals_path: Path,
                 patch_size=(128, 128, 128),
                 min_labeled_ratio=0.8,
                 xy_offsets=[1, 3, 6],
                 z_offsets=[1, 3, 6],
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
        self.cache_file = cache_file
        self.use_cache = use_cache

        estimated_mb = self._estimate_single_patch_memory()
        print(f"[INFO] Approx. single-patch GPU memory usage (float32): {estimated_mb:.2f} MB")

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

        # set up our affinity transform
        self.affinity_transform = LabelToAffinities(
            offsets=xy_offsets,
            z_offsets=z_offsets,
            append_label=False,
            aggregate_affinities=False
        )
        print("AffinityTransform parameters:")
        print(f"  z_offsets = {self.affinity_transform.z_offsets}")
        print(f"  append_label = {self.affinity_transform.append_label}")
        print(f"  aggregate_affinities = {self.affinity_transform.aggregate_affinities}")

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

    def _estimate_single_patch_memory(self):
        """
        Estimate (in MB) the memory usage for a single patch
        (image + label + normals + affinity maps) in float32.

        This is a naive calculation that assumes:
          - image is (1, Z, Y, X)
          - label is (1, Z, Y, X)
          - normals is (3, Z, Y, X)
          - affinity is (n_offsets, Z, Y, X)
          - float32 => 4 bytes per element
        """
        bytes_per_float = 4  # float32
        Z, Y, X = self.patch_size

        # --- Image: shape = (1, Z, Y, X)
        image_size = 1 * Z * Y * X

        # --- Label: shape = (1, Z, Y, X)
        label_size = 1 * Z * Y * X

        # --- Normals: shape = (3, Z, Y, X)
        normals_size = 3 * Z * Y * X

        # --- Affinity: shape = (#_offsets, Z, Y, X)
        # For example, if we have xy_offsets=[1,3,6] => 3 offsets in xy
        # and z_offsets=[1,3,6] => 3 offsets in z,
        # total offsets = 6 => shape = (6, Z, Y, X)
        num_offsets = len(self.xy_offsets) + len(self.z_offsets)
        affinity_size = num_offsets * Z * Y * X

        total_elements = image_size + label_size + normals_size + affinity_size
        total_bytes = total_elements * bytes_per_float
        total_mb = total_bytes / (1024 ** 2)

        return total_mb

    def _find_valid_patches(self, num_workers=12):
        if self._is_single_volume:
            sheet_label = self.sheet_label_array
        else:
            # Use the 0-th volume as reference
            sheet_label = self.sheet_label_array[0]

        print(f"finding bounding box for all labels in reference volume...")
        # find the bounding box that contains all labels
        bb = find_label_bounding_box(sheet_label)
        minz, maxz, miny, maxy, minx, maxx = bb
        print("Bounding box of labeled region:", bb)

        pD, pH, pW = self.patch_size

        # generate all potential (Z, Y, X) positions
        all_positions = []
        for z in range(minz, maxz - pD + 2, pD // 2):
            for y in range(miny, maxy - pH + 2, pH // 2):
                for x in range(minx, maxx - pW + 2, pW // 2):
                    all_positions.append((z, y, x))

        total_positions = len(all_positions)
        print("Number of potential patch positions:", total_positions)

        # chunk out the checking
        chunk_size = max(1, total_positions // (num_workers * 2))
        position_chunks = list(_chunker(all_positions, chunk_size))

        valid_positions_ref = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            async_results = []
            for chunk in position_chunks:
                async_results.append(pool.apply_async(
                    _check_patch_chunk,
                    (chunk, sheet_label, self.patch_size, self.min_labeled_ratio)
                ))

            for r in tqdm(async_results, desc="Checking reference patches", total=len(async_results)):
                valid_positions_ref.extend(r.get())  # gather results

        # if single volume just store them, otherwise replicate for each volume
        self.valid_patches = []
        if self._is_single_volume:
            for (z, y, x) in valid_positions_ref:
                self.valid_patches.append({
                    'volume_idx': 0,
                    'start_pos': [z, y, x]  # store as list for JSON
                })
        else:
            for volume_idx in range(self.n_volumes):
                for (z, y, x) in valid_positions_ref:
                    self.valid_patches.append({
                        'volume_idx': volume_idx,
                        'start_pos': [z, y, x]
                    })

        print(f"Found {len(valid_positions_ref)} valid patches in the reference volume.")
        print(f"Replicated to total of {len(self.valid_patches)} valid patches "
              f"across {self.n_volumes} volume(s).")


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
        if volume_data.dtype == np.uint8:
            images /= 255.0
        elif volume_data.dtype == np.uint16:
            images /= 65535.0

        images = self.standardizer(images)

        # --- sheet data  ---
        if self._is_single_volume:
            sheet_label = self.sheet_label_array[patch_slice].astype(np.float32)
        else:
            sheet_label = self.sheet_label_array[volume_idx][patch_slice].astype(np.float32)

        # normalize to 0, 1
        sheet_label /= 255.0

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

        # ---- Affinity maps ------
        affinity_maps = self.affinity_transform(sheet_label)

        # add a dimension just so i can stop trying to deal with 3d v 4d in transforms
        images = images[np.newaxis, ...]  # (Z, Y, X) -> (1, Z, Y, X)
        sheet_label = sheet_label[np.newaxis, ...]  # (Z, Y, X) -> (1, Z, Y, X)

        # ------ augs -------
        if self.transforms:
            data_dict = {
                'image': images,
                'sheet_label': sheet_label,
                'normals': normals,
                'affinity': affinity_maps
            }
            data_dict = self.transforms(data_dict)
            images = data_dict['image']
            sheet_label = data_dict['sheet_label']
            normals = data_dict['normals']
            affinity_maps = data_dict['affinity']

        # convert to tensors
        images = torch.from_numpy(np.ascontiguousarray(images))
        sheet_label = torch.from_numpy(np.ascontiguousarray(sheet_label))
        normals = torch.from_numpy(np.ascontiguousarray(normals))
        affinity_maps = torch.from_numpy(np.ascontiguousarray(affinity_maps))

        # now doing this before augs
        # images = images.unsqueeze(0)  # (Z, Y, X) => (1, Z, Y, X) or (1, Z, Y, X, C)
        # sheet_label = sheet_label.unsqueeze(0)  # (Z, Y, X) => (1, Z, Y, X)


        return images, sheet_label, normals, affinity_maps

    def close(self):
        """Close the Zarr arrays."""
        self.volume_array.store.close()
        self.sheet_label_array.store.close()
        self.normals_array.store.close()
