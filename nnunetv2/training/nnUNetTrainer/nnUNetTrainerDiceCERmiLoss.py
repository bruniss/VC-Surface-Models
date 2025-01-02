import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import RMI_DC_and_CE_loss
import numpy as np


class nnUNetTrainerDiceCERmiLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        ignore_index = self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100

        loss = RMI_DC_and_CE_loss(
            soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={'weight': None, 'ignore_index': ignore_index},
            rmi_kwargs={'num_classes': self.label_manager.num_segmentation_heads, 'rmi_radius': 5},
            ignore_label=ignore_index,
            weight_ce=0.2,
            weight_dice=0.4,
            weight_rmi=0.4,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss