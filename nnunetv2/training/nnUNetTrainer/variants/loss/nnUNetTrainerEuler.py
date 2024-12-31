from typing import Union, Tuple, List

import torch
from torch import nn
import os
from nnunetv2.training.loss.cldice import soft_cldice
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class CustomTrainerEuler(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """
        Custom nnUNet trainer that uses CL_and_DC_and_BCE_loss
        """
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def _build_loss(self):


        loss = CL_and_DC_and_BCE_loss(
            bce_kwargs={},
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': True,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            use_ignore_label=self.label_manager.ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss,
            cldice_version="skel"  # You can modify this to use different versions
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # weights for deep supervision - same as original implementation
            weights = [1 / (2 ** i) for i in range(len(deep_supervision_scales))]

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = torch.tensor(weights) / torch.tensor(weights).sum()

            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class CL_and_DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss, cldice_version="euler"):
        """
        Combined loss that uses Binary Cross Entropy, Dice Loss, and Centerline Dice Loss.
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        Args:
            bce_kwargs (dict): Arguments for BCEWithLogitsLoss
            soft_dice_kwargs (dict): Arguments for Dice Loss
            weight_ce (float): Weight for Cross Entropy loss
            weight_dice (float): Weight for Dice loss
            use_ignore_label (bool): Whether to use ignore label
            dice_class: Class to use for Dice loss computation
            cldice_version (str): Version of centerline dice to use ("legacy", "fast", "skel", "euler")
        """
        super().__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        # Get weights from environment variables if set, otherwise use defaults
        self.weight_ce = float(os.getenv('WEIGHT_CE', weight_ce))
        self.weight_dice = float(os.getenv('WEIGHT_DICE', weight_dice))
        self.weight_cl = float(os.getenv('WEIGHT_CL', 0.002))

        self.use_ignore_label = use_ignore_label
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

        # Initialize appropriate centerline dice version
        if cldice_version == "fast":
            self.cl = soft_cldice()
        elif cldice_version == "skel":
            self.cl = soft_cldice()
        elif cldice_version == "euler":
            self.cl = soft_cldice(skel_strat="EulerCharacteristic")


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Forward pass of the loss function.

        Args:
            net_output (torch.Tensor): Network output logits
            target (torch.Tensor): Ground truth target (one-hot encoded)

        Returns:
            torch.Tensor: Combined weighted loss
        """
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        # Calculate Dice loss
        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)

        # Calculate BCE loss
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)

        # Calculate Centerline Dice loss
        cl_loss = self.cl(y_pred=net_output, y_true=target)

        # Combine losses with their respective weights
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cl * cl_loss
        return result