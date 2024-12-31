from typing import Tuple, Union, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans


class nnUNetTrainerUMambaBot(nnUNetTrainer):
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(self.configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(self.plans_manager, self.dataset_json, self.configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(self.configuration_manager.patch_size) == 3:
            model = get_umamba_bot_3d_from_plans(self.plans_manager, self.dataset_json, self.configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        print("UMambaBot: {}".format(model))

        return model
