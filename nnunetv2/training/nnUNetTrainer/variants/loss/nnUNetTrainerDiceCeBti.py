from nnunetv2.training.loss.compound_bti_loss import DC_and_CE_and_BTI_Loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerDiceCeBti(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("this trainer does not support region-based training")

        patch_size = self.configuration_manager.patch_size

        dim = len(patch_size)
        if dim == 3:
            connectivity = 26
            lambda_ti = 1e-6
        else:
            connectivity = 8
            lambda_ti = 1e-4

        # classes that should touch go in inclusion, those that shuold not go in exclusion
        inclusion_list = []
        exclusion_list = [[1, 1]]

        # distance to try and maintain between structures / classes
        min_thick = 1

        loss = DC_and_CE_and_BTI_Loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-3,
                'do_bg': False,
                'ddp': self.is_ddp
            },
            ce_kwargs={},
            ti_kwargs={
                'dim': len(self.configuration_manager.patch_size),
                'connectivity': connectivity,
                'inclusion': inclusion_list,
                'exclusion': exclusion_list,
                'min_thick': min_thick
            },
            weight_ce=1,
            weight_dice=1,
            weight_ti=lambda_ti,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        return loss