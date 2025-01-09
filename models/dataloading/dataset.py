# this is very heavily borrowed from discord user @Mojonero, who kindly shared his s2 starter here: https://discord.com/channels/1079907749569237093/1204133327083147264/1204133327083147264
from typing import Tuple, Union, List
import zarr
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from skimage.morphology import dilation, ball
from pytorch3dunet.augment.transforms import (Compose, LabelToAffinities, Standardize,
                                              RandomFlip, RandomRotate90)

from models.helpers import (_find_valid_patches)

from models.transforms.normalization import ct_normalize

NORMALIZATION_FUNCS = {
            "ct_normalization": ct_normalize
        }

class ZarrSegmentationDataset3D(Dataset):
    def __init__(self,
                 data_path: Path,
                 targets: dict,
                 patch_size=(128, 128, 128),
                 min_labeled_ratio=0.9,
                 min_bbox_percent=95,
                 normalization=ct_normalize,
                 dilate_label=False,
                 transforms=None,
                 use_cache=True,
                 cache_file: Path = "./valid_cache.json",
                 ):

        self.data_path = data_path
        self.patch_size = patch_size
        self.min_labeled_ratio = min_labeled_ratio
        self.bbox_threshold = min_bbox_percent
        self.normalization = normalization
        self.normalization_func = NORMALIZATION_FUNCS[self.normalization]
        self.dilate_label = dilate_label
        self.transforms_list = transforms
        self.cache_file = cache_file
        self.use_cache = use_cache


        self.input_array = zarr.open(str(data_path), mode='r')

        self.target_arrays = {}
        for target_name, target_info in targets.items():
            real_path = target_info["dataset_path"]  # Extract the real path string
            print(f"Opening target array '{target_name}' from {real_path}...")

            z = zarr.open(str(real_path), mode='r')  # Now this is a proper path
            self.target_arrays[target_name] = z


        # data input volume should be (Z, Y, X) or (Z, Y, X, C)
        self.ndim = self.input_array.ndim

        # if cache file exists, load valid patches.
        # otherwise, compute them and optionally write to cache.
        reference_array = self.target_arrays["sheet"]
        self.valid_patches = []
        if self.use_cache and self.cache_file is not None and self.cache_file.exists():
            print(f"Cache file found at {self.cache_file}. Loading valid patches...")
            with open(self.cache_file, 'r') as f:
                # Note: these will load back as lists, so 'start_pos' will be a list, etc.
                self.valid_patches = json.load(f)
            print(f"Loaded {len(self.valid_patches)} valid patches from cache.")
        else:
            # compute valid patches
            self.valid_patches = _find_valid_patches(reference_array,
                                 self.patch_size,
                                 self.bbox_threshold,
                                 self.min_labeled_ratio)
            print(f"Found {len(self.valid_patches)} valid patches.")

            # write to cache if use_cache
            if self.use_cache and self.cache_file is not None:
                print(f"Saving valid patches to {self.cache_file}...")
                with open(self.cache_file, 'w') as f:
                    json.dump(self.valid_patches, f)


    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_info = self.valid_patches[idx]
        z0, y0, x0 = patch_info['start_pos']
        dz, dy, dx = self.patch_size

        patch_slice = np.s_[z0:z0+dz, y0:y0+dy, x0:x0+dx] # get the corner indices, expand by patch size

        # --- input data --- #

        input_data = self.input_array[patch_slice]
        images = input_data.astype(np.float32)
        images /= 255.0
        images = self.normalization_func(
            images,
            clip_min=43.0,
            clip_max=240.0,
            global_mean=129.9852752685547,
            global_std=44.020145416259766
        )

        # --- target data  --- #
        targets_data = {}
        for target_name, z_arr in self.target_arrays.items():
            target_patch = z_arr[patch_slice]

            if target_name.lower() in ("normals", "normal"):
                # Known to be uint16 -> scaled to [-1,1]
                target_patch = target_patch.astype(np.float32)
                target_patch /= 32767.5
                target_patch -= 1.0
                target_patch = target_patch.transpose(3, 0, 1, 2).copy()

            else:
                target_patch = target_patch.astype(np.float32)

                # If it's 8-bit in [0,255], scale
                if z_arr.dtype == np.uint8:
                    target_patch /= 255.0

                # If it's 16-bit in [0,65535], scale
                elif z_arr.dtype == np.uint16:
                    target_patch /= 65535.0


                # check for dilation
                if (target_name.lower() == "sheet" or target_name.lower() == "sheet_label") and self.dilate_label:
                    target_patch = (target_patch > 0.5).astype(np.float32)
                    target_patch = dilation(target_patch, ball(radius=1))
                    target_patch = (target_patch > 0.5).astype(np.float32)

            targets_data[target_name] = target_patch

        data_dict = {'image': images}
        data_dict.update(targets_data)

        # ---- augmentation ----- #
        # data is still numpy array, we have not added an addtl channel, shape should be Z,Y,X or C, Z, Y, X
        if self.transforms_list is not None:
            data_dict = self.transforms_list(data_dict)


        # ------ convert everything to tensors ----- #

        # image data to tensor Z, X, Y -> C, Z, X, Y
        if data_dict['image'].ndim == 3:
            data_dict['image'] = data_dict['image'][None, ...]  # add channel dimension
        data_dict['image'] = torch.from_numpy(np.ascontiguousarray(data_dict['image']))

        # target data to tensor , if only 3 dims, add one -- Z, Y, X -> C, Z, Y, X
        for target_name in targets_data.keys():
            t_patch = data_dict[target_name]
            if t_patch.ndim == 3:
                # e.g. sheet_label is (Z, Y, X) => add a channel dimension
                t_patch = t_patch[None, ...]
            t_patch = torch.from_numpy(np.ascontiguousarray(t_patch))
            data_dict[target_name] = t_patch

        return data_dict

    def close(self):
        """Close the Zarr arrays if needed."""
        self.input_array.store.close()
        for t_name, t_arr in self.target_arrays.items():
            t_arr.store.close()
