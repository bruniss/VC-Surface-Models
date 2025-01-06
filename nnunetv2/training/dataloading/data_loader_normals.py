import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

class nnUNetDataLoader3DSkel(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        normal_data = np.zeros(self.normal_shape, dtype=np.float32)  # Add this for normals
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            # Load both regular data and normal data
            data, normal, seg, properties = self._data.load_case(i)  # You'll need to modify load_case
            case_properties.append(properties)

            # Process regular data as before
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # Process regular data
            this_slice = tuple(
                [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            # Process normal data
            if normal.dtype == np.uint16:
                normal = (normal.astype(np.float32) / 32767.5) - 1
            normal = normal[this_slice]

            # Process segmentation
            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            # Padding
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            # normal_data[j] = np.pad(normal, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    normal_data = torch.from_numpy(normal_data).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)

                    images = []
                    normals = []
                    segs = []


                    for b in range(self.batch_size):
                        # You might want separate transforms for images and normals
                        tmp = self.transforms(**{
                            'image': data_all[b],
                            'normal': normal_data[b],
                            'segmentation': seg_all[b]
                        })
                        images.append(tmp['image'])
                        normals.append(tmp['normal'])
                        segs.append(tmp['segmentation'])
                        skels.append(tmp['skel'])

                    data_all = torch.stack(images)
                    normal_data = torch.stack(normals)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                        skel_all = [torch.stack([s[i] for s in skels]) for i in range(len(skels[0]))]
                    else:
                        seg_all = torch.stack(segs)
                        skel_all = torch.stack(skels)
                    del segs, images, skels, normals

        return {
            'data': data_all,
            'normal': normal_data,  # Add normals to return dict
            'target': seg_all,
            'skel': skel_all,
            'keys': selected_keys
        }