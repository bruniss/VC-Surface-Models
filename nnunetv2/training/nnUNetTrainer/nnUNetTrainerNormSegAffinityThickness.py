from typing import Tuple, Union, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import torch
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans


class MultiTaskResidualEncoderUNet(ResidualEncoderUNet):
    def __init__(self, *args, num_normal_channels=3, **kwargs):
        super().__init__(*args, **kwargs)

        # Get encoder features for creating normal head
        features_of_last_stage = self.encoder.output_channels[-1]

        # Create normal prediction head
        self.normal_head = torch.nn.Sequential(
            torch.nn.Conv3d(features_of_last_stage, 64, 3, padding=1),
            get_matching_instancenorm(self.configuration_manager.norm_op),
            torch.nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            torch.nn.Conv3d(64, num_normal_channels, 1)
        )

    def forward(self, x):
        skips = self.encoder(x)
        segmentation = self.decoder(skips)

        # Get encoder features for normal prediction
        normals = self.normal_head(skips[-1])
        normals = torch.nn.functional.normalize(normals, dim=1)  # Normalize to unit vectors

        if self.deep_supervision:
            return segmentation, normals
        else:
            if isinstance(segmentation, (tuple, list)):
                segmentation = segmentation[0]
            return segmentation, normals

class nnUNetTrainerNormSegAffThickness(nnUNetTrainer):


    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = MultiTaskResidualEncoderUNet(
            input_channels=self.num_input_channels,
            n_stages=self.configuration_manager.n_stages,
            features_per_stage=self.configuration_manager.features_per_stage,
            conv_op=self.configuration_manager.conv_op,
            kernel_sizes=self.configuration_manager.kernel_sizes,
            strides=self.configuration_manager.strides,
            n_blocks_per_stage=self.configuration_manager.n_blocks_per_stage,
            num_classes=self.label_manager.num_segmentation_heads,
            n_conv_per_stage_decoder=self.configuration_manager.n_conv_per_stage_decoder,
            conv_bias=self.configuration_manager.conv_bias,
            norm_op=self.configuration_manager.norm_op,
            norm_op_kwargs=self.configuration_manager.norm_op_kwargs,
            dropout_op=self.configuration_manager.dropout_op,
            nonlin=self.configuration_manager.nonlin,
            nonlin_kwargs=self.configuration_manager.nonlin_kwargs,
            deep_supervision=self.enable_deep_supervision,
            num_normal_channels=3
        )

        return model

    def _build_loss(self):
        seg_loss = super()._build_loss()
        normal_loss = torch.nn.MSELoss()

        def combined_loss(pred, target):
            seg_pred, normal_pred = pred
            seg_target, normal_target = target

            # Get losses
            s_loss = seg_loss(seg_pred, seg_target)
            n_loss = normal_loss(normal_pred, normal_target)

            # Combine with weighting
            return s_loss + 0.1 * n_loss

        return combined_loss
