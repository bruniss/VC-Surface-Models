# more or less copied directly from https://github.com/MIC-DKFZ/batchgeneratorsv2/blob/master/batchgeneratorsv2/transforms/base/basic_transform.py

import abc
import torch

class BasicTransform(abc.ABC):
    """
    Transforms are applied to each sample individually, can specify image only , label only, etc as long as keyword exists

    We expect (C, Z, Y, X) shaped inputs

    """
    def __init__(self):
        pass

    def __call__(self, **data_dict) -> dict:
        params = self.get_parameters(**data_dict)
        return self.apply(data_dict, **params)

    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)

        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)

        # this will be for vector/vectorlike inputs (im adapting it for normals)
        # this should apply geometric transformations to vectors such that they transform to match the image
        if data_dict.get('regression_target') is not None:
            data_dict['regression_target'] = self._apply_to_segmentation(data_dict['regression_target'], **params)


        return data_dict

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        pass

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_keypoints(self, keypoints, **params):
        pass

    def _apply_to_bbox(self, bbox, **params):
        pass

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class ImageOnlyTransform(BasicTransform):
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)
        return data_dict


class SegOnlyTransform(BasicTransform):
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)
        return data_dict


if __name__ == '__main__':
    pass