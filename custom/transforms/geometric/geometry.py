import random
import torch
import numpy as np
from custom.transforms.base_transform import BasicTransform
from typing import Tuple


class RandomRotate90(BasicTransform):
    """
    Randomly applies a 90°, 180°, or 270° rotation to the image, segmentation, and regression targets.
    Supports rotations in the XY, XZ, or YZ planes.
    """
    def __init__(self, allowed_axes: Tuple[Tuple[int, int], ...]):
        """
        Args:
            allowed_axes (Tuple[Tuple[int, int], ...]): Allowed rotation planes, e.g., [(1, 2), (0, 2), (0, 1)].
        """
        super().__init__()
        self.allowed_axes = allowed_axes
        self.rot_axes = random.choice(allowed_axes)

        rotations = [90, 180, 270]
        self.rot_angle = random.choice(rotations)

        # Determine rotation count for torch.rot90
        self.rot_count = rotations.index(self.rot_angle) + 1

    def get_parameters(self, **data_dict) -> dict:
        return {
            'rot_axes': self.rot_axes,
            'rot_angle': self.rot_angle
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return torch.rot90(img, self.rot_count, self.rot_axes)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return torch.rot90(segmentation, self.rot_count, self.rot_axes)

    def _apply_to_regr_target(self, regression_target: torch.Tensor, **params) -> torch.Tensor:
        """
        Applies the rotation to the regression target (e.g., normals).
        Adjusts vector components to match the rotation.
        """
        assert regression_target.shape[0] == 3, "Regression target must have 3 channels for (x, y, z) components."

        # Rotate the spatial dimensions
        rotated_target = torch.rot90(regression_target, self.rot_count, self.rot_axes)

        # Adjust the vector components to match the rotation
        if self.rot_axes == (1, 2):  # Rotate in the XY plane (around Z-axis)
            rotated_target = rotated_target.clone()
            rotated_target[0], rotated_target[1] = -rotated_target[1], rotated_target[0]
        elif self.rot_axes == (0, 2):  # Rotate in the XZ plane (around Y-axis)
            rotated_target = rotated_target.clone()
            rotated_target[0], rotated_target[2] = rotated_target[2], -rotated_target[0]
        elif self.rot_axes == (0, 1):  # Rotate in the YZ plane (around X-axis)
            rotated_target = rotated_target.clone()
            rotated_target[1], rotated_target[2] = -rotated_target[2], rotated_target[1]

        return rotated_target

    def _apply_to_bbox(self, bbox, **params):
        """
        Placeholder for applying rotation to bounding boxes.
        """
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        """
        Placeholder for applying rotation to keypoints.
        """
        raise NotImplementedError
