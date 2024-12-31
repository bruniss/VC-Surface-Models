
"""
Your loss function must inherit from nn.Module
The forward method should take net_output and target as inputs
Handle ignore labels appropriately if your loss supports them
For compound losses, make sure to properly weight and combine the individual loss components
In the trainer class, always handle deep supervision if it's enabled
Make sure to properly handle batch dice and DDP (distributed data parallel) settings if relevant
"""

"""
import torch.nn as nn
class YourCustomLoss(nn.Module):
    def __init__(self, your_params):
        super(YourCustomLoss, self).__init__()
        # Initialize your loss parameters here
        self.your_params = your_params

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # Implement your loss computation here
        # net_output: network predictions
        # target: ground truth
        return loss_value

class Custom_Compound_Loss(nn.Module):
    def __init__(self, loss1_kwargs, loss2_kwargs, weight1=1, weight2=1, ignore_label=None):
        super(Custom_Compound_Loss, self).__init__()

        # Setup ignore label if needed
        if ignore_label is not None:
            loss1_kwargs['ignore_index'] = ignore_label

        self.weight1 = weight1
        self.weight2 = weight2
        self.ignore_label = ignore_label

        # Initialize your loss components
        self.loss1 = FirstLoss(**loss1_kwargs)
        self.loss2 = SecondLoss(**loss2_kwargs)

    def forward(self, net_output, target):
        # Handle ignore label if present
        if self.ignore_label is not None:
            mask = (target != self.ignore_label).bool()
            target_modified = torch.clone(target)
            target_modified[target == self.ignore_label] = 0
        else:
            target_modified = target
            mask = None

        # Compute individual losses
        loss1_val = self.loss1(net_output, target_modified)
        loss2_val = self.loss2(net_output, target_modified)

        # Combine losses
        return self.weight1 * loss1_val + self.weight2 * loss2_val

"""