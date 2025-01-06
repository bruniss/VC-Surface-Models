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

class ZarrSegmentationDataset3D(Dataset):
    def __init__(self,
                 volume_path: Path,
                 sheet_label_path: Path,
                 normals_path: Path,
                 patch_size=(128, 128, 128),
                 min_labeled_ratio=0.8,
                 xy_offsets=[1, 3, 6],
                 z_offsets=[1, 3, 6],
                 transforms_list=None):

        self.volume_path = volume_path
        self.sheet_label_path = sheet_label_path
        self.normals_path = normals_path
        self.patch_size = patch_size
        self.min_labeled_ratio = min_labeled_ratio
        self.xy_offsets = xy_offsets
        self.z_offsets = z_offsets

        # open up our zarr arrays
        self.volume_array = zarr.open(str(volume_path), mode='r')
        self.sheet_label_array = zarr.open(str(sheet_label_path), mode='r')
        self.normals_array = zarr.open(str(normals_path), mode='r')

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

        # gather transforms with their targets and probabilities
        if transforms_list:
            targeted_transforms = [
                TargetedTransform(transform, targets)
                for transform, targets in transforms_list
            ]
            self.transforms = Compose(targeted_transforms)
        else:
            self.transforms = None

        # check if we have single-volume or multi-volume
        # e.g., single-volume should be (D, H, W) or (D, H, W, C)
        # multi-volume mask should be (N, D, H, W) or (N, D, H, W, C)
        self.ndim = self.sheet_label_array.ndim

        if self.ndim == 3 or self.ndim == 4:
            # single volume
            self._is_single_volume = True
            # shape is either (D, H, W) or (D, H, W, C)
            # We interpret "volume_shape" as the full shape minus any channels
            # but for mask, presumably it's just (D, H, W).
            self._volume_depth = self.sheet_label_array.shape[0]
            self._volume_height = self.sheet_label_array.shape[1]
            self._volume_width = self.sheet_label_array.shape[2]
            # We'll store only one 'volume_idx' = 0
            self.n_volumes = 1
        else:
            # Multi-volume scenario
            self._is_single_volume = False
            # shape is (N, D, H, W) or (N, D, H, W, C)
            self.n_volumes = self.sheet_label_array.shape[0]
            self._volume_depth = self.sheet_label_array.shape[1]
            self._volume_height = self.sheet_label_array.shape[2]
            self._volume_width = self.sheet_label_array.shape[3]

        # get list of valid patches
        self.valid_patches = []
        self._find_valid_patches()

    def _find_valid_patches(self):
        # for each volume_idx in range(self.n_volumes) , this is because i have some w/ more than one cause i wrote it bad
        for volume_idx in range(self.n_volumes):
            # load the entire sheet label for that volume to find valid patches
            if self._is_single_volume:
                # there's only one volume, so volume_idx=0 => we have the entire array
                sheet_label = self.sheet_label_array
            else:
                sheet_label = self.sheet_label_array[volume_idx]

            # sheet_label now has shape (D, H, W) or (D, H, W, C),
            # but presumably for a mask it's (D, H, W).
            volume_shape = sheet_label.shape  # (D, H, W)

            patch_d, patch_h, patch_w = self.patch_size

            # slice window 1/2 of patch size, some overlap
            for z in range(0, volume_shape[0] - patch_d + 1, patch_d // 2):
                for y in range(0, volume_shape[1] - patch_h + 1, patch_h // 2):
                    for x in range(0, volume_shape[2] - patch_w + 1, patch_w // 2):
                        patch = sheet_label[z:z + patch_d,
                                           y:y + patch_h,
                                           x:x + patch_w]

                        # check to see if patch contains desired amt of labeled data, in percentages
                        if np.sum(patch > 0) / patch.size >= self.min_labeled_ratio:
                            self.valid_patches.append({
                                'volume_idx': volume_idx,  # 0 if single volume
                                'start_pos': (z, y, x)
                            })

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

        # --- Normals data ---
        if self._is_single_volume:
            normals = self.normals_array[patch_slice].astype(np.float32)
        else:
            normals = self.normals_array[volume_idx][patch_slice].astype(np.float32)

        # normalize the....normals....
        magnitudes = np.linalg.norm(normals, axis=-1, keepdims=True)
        valid = magnitudes > 0
        normals[valid] /= magnitudes[valid]

        # ---- Affinity maps ------
        affinity_maps = self.affinity_transform(sheet_label)

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
        images = torch.from_numpy(images)       # shape => (D, H, W) or (D, H, W, C)
        sheet_label = torch.from_numpy(sheet_label)
        normals = torch.from_numpy(normals)
        affinity_maps = torch.from_numpy(affinity_maps)

        images = images.unsqueeze(0)  # => (1, D, H, W) or (1, D, H, W, C)
        sheet_label = sheet_label.unsqueeze(0)  # => (1, D, H, W)
        normals = normals.permute(3, 0, 1, 2) # (D, H, W, 3) => (3, D, H, W)

        return images, sheet_label, normals, affinity_maps

    def close(self):
        """Close the Zarr arrays."""
        self.volume_array.store.close()
        self.sheet_label_array.store.close()
        self.normals_array.store.close()
