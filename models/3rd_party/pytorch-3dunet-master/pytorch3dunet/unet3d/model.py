import torch.nn as nn

from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level

def get_activation_module(activation_str: str):
    """
    Simple helper that returns an nn.Module (e.g. nn.Sigmoid) or None
    if activation_str == 'none'
    """
    if activation_str.lower() == "none":
        return None
    elif activation_str.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation_str.lower() == "softmax":
        return nn.Softmax(dim=1)
    # Add others if you like
    else:
        return None  # default is None, or raise an error


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                        conv_padding, conv_upscale, dropout_prob,
                                        layer_order, num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
                                        layer_order, num_groups, upsample, dropout_prob,
                                        is3d)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=True)


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=True)


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on 
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               conv_upscale=conv_upscale,
                                               upsample=upsample,
                                               dropout_prob=dropout_prob,
                                               is3d=True)


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=False)


class ResidualUNet2D(AbstractUNet):
    """
    Residual 2DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet2D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=False)



class MultiTaskResidualUNetSE3D(ResidualUNetSE3D):
    """
    A U-Net style architecture with a shared encoder and multiple decoders,
    one for each task in a dictionary of tasks.
    """

    def __init__(
        self,
        in_channels: int,
        tasks: dict, # example: tasks = {"sheet": 1, "normals": 3, "affinities": 6}
        final_sigmoid=True,
        f_maps=[32, 64, 128, 256, 320, 320, 320],
        layer_order='gcr',
        num_groups=8,
        num_levels=7,
        conv_padding=1,
        conv_upscale=2,
        upsample='default',
        dropout_prob=0.1
    ):
        """
        Args:
            in_channels (int): Number of input channels (e.g. 1 for grayscale).
            tasks (dict): Dictionary of {task_name: out_channels} for each task.
            final_sigmoid (bool): If True, apply sigmoid to the 'sheet' and 'affinities' tasks (or similarly).
            ...
        """

        # Initialize the parent (which constructs the shared encoder)
        # We pass out_channels=1 as a dummy, since we won't use the parent's
        # final conv. We'll have our own final conv for each task.
        super().__init__(
            in_channels=in_channels,
            out_channels=1,  # dummy
            final_sigmoid=False,  # we won't use the parent's final sigmoid
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=True,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob
        )

        self.tasks = tasks
        self.f_maps = f_maps

        # These ModuleDicts will store decoders and final_convs for each task
        self.task_decoders = nn.ModuleDict()
        self.task_final_convs = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()

        # For each task, create a decoder path and final conv
        for task_name, task_info in self.tasks.items():
            out_channels = task_info["channels"]
            activation_str = task_info.get("activation", "sigmoid")  # default is 'sigmoid' if not specified

            # Create the decoder path for this task
            self.task_decoders[task_name] = create_decoders(
                f_maps=self.f_maps,
                basic_module=ResNetBlockSE,
                conv_kernel_size=3,
                conv_padding=conv_padding,
                layer_order=layer_order,
                num_groups=num_groups,
                upsample=upsample,
                dropout_prob=dropout_prob,
                is3d=True
            )

            # Create the final 1x1 conv
            self.task_final_convs[task_name] = nn.Conv3d(self.f_maps[0], out_channels, kernel_size=1)

            activation_module = get_activation_module(activation_str)
            self.task_activations[task_name] = activation_module

    def forward(self, x):
        # Shared encoder
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        # Remove the deepest features from skip connections
        encoders_features = encoders_features[1:]

        # Dictionary of outputs
        results = {}

        # For each task, run the decoder + final conv + activation
        for task_name, decoder_path in self.task_decoders.items():
            task_x = x  # deepest features from encoder
            # decode with skip connections
            for decoder, skip_features in zip(decoder_path, encoders_features):
                task_x = decoder(skip_features, task_x)

            # final 1×1 conv
            task_x = self.task_final_convs[task_name](task_x)

            # apply activation only if not None
            if self.task_activations[task_name] is not None:
                task_x = self.task_activations[task_name](task_x)

            results[task_name] = task_x

        return results


def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)
