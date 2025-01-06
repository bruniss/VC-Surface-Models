import importlib
import random

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m

def reorder_channels(m, reorder_dict):
    """
    Reorders (or swaps) channels in 'm' according to 'reorder_dict'.
    m has shape (C, D, H, W).
    reorder_dict is a mapping old_channel -> new_channel.
    """
    new_m = np.zeros_like(m)
    for old_ch, new_ch in reorder_dict.items():
        new_m[new_ch] = m[old_ch]
    return new_m

class RandomFlip:
    """
    Flip along the spatial dimensions. If shape=(C,D,H,W), axes=(1,2,3).
    If shape=(D,H,W), axes=(0,1,2).

    Optionally:
      - is_normals: negate the appropriate normal channel(s)
      - is_affinities: swap the +/- offset channels
    """
    def __init__(self,
                 random_state,
                 axis_prob=0.5,
                 is_normals=False,
                 is_affinities=False,
                 **kwargs):
        self.random_state = random_state
        self.axis_prob = axis_prob
        self.is_normals = is_normals
        self.is_affinities = is_affinities

    def __call__(self, m):
        # Decide which axes are "spatial" based on ndim
        if m.ndim == 4:
            # shape=(C,D,H,W) => spatial axes are (1,2,3)
            spatial_axes = (1, 2, 3)
        elif m.ndim == 3:
            # shape=(D,H,W) => spatial axes are (0,1,2)
            spatial_axes = (0, 1, 2)
        else:
            raise ValueError(f"Unsupported ndim={m.ndim}. Expect 3 or 4.")

        for axis in spatial_axes:
            if self.random_state.uniform() < self.axis_prob:
                if m.ndim == 4:

                    m = np.flip(m, axis=axis)

                    # If we have 3-channel normals (nx,ny,nz) in shape=(3,D,H,W):
                    if self.is_normals and m.shape[0] == 3:
                        # Map which normal channel to negate for each flip axis
                        # If axis=1 => flipping along "D" => negate channel=2 (nz)
                        # If axis=2 => flipping along "H" => negate channel=1 (ny)
                        # If axis=3 => flipping along "W" => negate channel=0 (nx)
                        flip_map_normals = {1: 2, 2: 1, 3: 0}
                        ch = flip_map_normals[axis]
                        m[ch] *= -1.0

                    # If we have 6-channel affinity in shape=(6,D,H,W):
                    if self.is_affinities and m.shape[0] == 6:
                        # Suppose channels = {0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z}
                        if axis == 3:
                            # flipping along W => swap +x <-> -x
                            swap_dict = {0:1, 1:0, 2:2, 3:3, 4:4, 5:5}
                            m = reorder_channels(m, swap_dict)
                        elif axis == 2:
                            # flipping along H => swap +y <-> -y
                            swap_dict = {0:0, 1:1, 2:3, 3:2, 4:4, 5:5}
                            m = reorder_channels(m, swap_dict)
                        elif axis == 1:
                            # flipping along D => swap +z <-> -z
                            swap_dict = {0:0, 1:1, 2:2, 3:3, 4:5, 5:4}
                            m = reorder_channels(m, swap_dict)

                else:
                    # m.ndim==3 => just flip directly
                    m = np.flip(m, axis=axis)

        return m

class RandomRotate90:
    def __init__(self, random_state, execution_probability=0.3, is_normals=False, is_affinities=False):
        self.random_state = random_state
        self.execution_probability = execution_probability
        self.is_normals = is_normals
        self.is_affinities = is_affinities

    def __call__(self, m):

        if self.random_state.uniform() > self.execution_probability:
            return m

        k = self.random_state.randint(0, 4)  # 0, 1, 2, or 3 times 90° rotation

        if k == 0:
            return m

        if m.ndim == 4:
            # shape = (C, D, H, W): rotate around the last two dims => axes=(2,3)
            m = np.rot90(m, k, axes=(2, 3))

            # After rotation, reorder channels if needed:
            if self.is_normals and m.shape[0] == 3:
                # Standard z-axis rotation rules:
                # k=1 => (nx, ny) -> (-ny, nx)
                # k=2 => (nx, ny) -> (-nx, -ny)
                # k=3 => (nx, ny) -> (ny, -nx)
                if k == 1:
                    # swap x & y, then negate the new x
                    m = m[[1, 0, 2]]  # reorder channels
                    m[0] *= -1.0      # was m[1] originally
                elif k == 2:
                    m[0] *= -1.0
                    m[1] *= -1.0
                elif k == 3:
                    m = m[[1, 0, 2]]
                    m[1] *= -1.0

            if self.is_affinities and m.shape[0] == 6:
                # Suppose the channels are {0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z}
                # rotating in (H,W) ~ rotating in the xy-plane => apply channel swaps:
                # For example, here’s one typical way:
                aff_rot90z = {
                    1: {0:3, 1:2, 2:1, 3:0, 4:4, 5:5},  # 90°
                    2: {0:1, 1:0, 2:3, 3:2, 4:4, 5:5},  # 180°
                    3: {0:2, 1:3, 2:0, 3:1, 4:4, 5:5},  # 270°
                }
                reorder_dict = aff_rot90z[k]
                m = reorder_channels(m, reorder_dict)

        elif m.ndim == 3:
            # shape = (D, H, W): rotate around last two dims => axes=(1,2)
            m = np.rot90(m, k, axes=(1, 2))

        else:
            raise ValueError(f"Unsupported ndim={m.ndim}. Expect 3 or 4.")

        return m




class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=2000, sigma=50, execution_probability=0.1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m


class CropToFixed:
    def __init__(self, random_state, size=(256, 256), centered=False, **kwargs):
        self.random_state = random_state
        self.crop_y, self.crop_x = size
        self.centered = centered

    def __call__(self, m):
        def _padding(pad_total):
            half_total = pad_total // 2
            return (half_total, pad_total - half_total)

        def _rand_range_and_pad(crop_size, max_size):
            """
            Returns a tuple:
                max_value (int) for the corner dimension. The corner dimension is chosen as `self.random_state(max_value)`
                pad (int): padding in both directions; if crop_size is lt max_size the pad is 0
            """
            if crop_size < max_size:
                return max_size - crop_size, (0, 0)
            else:
                return 1, _padding(crop_size - max_size)

        def _start_and_pad(crop_size, max_size):
            if crop_size < max_size:
                return (max_size - crop_size) // 2, (0, 0)
            else:
                return 0, _padding(crop_size - max_size)

        assert m.ndim in (3, 4)
        if m.ndim == 3:
            _, y, x = m.shape
        else:
            _, _, y, x = m.shape

        if not self.centered:
            y_range, y_pad = _rand_range_and_pad(self.crop_y, y)
            x_range, x_pad = _rand_range_and_pad(self.crop_x, x)

            y_start = self.random_state.randint(y_range)
            x_start = self.random_state.randint(x_range)

        else:
            y_start, y_pad = _start_and_pad(self.crop_y, y)
            x_start, x_pad = _start_and_pad(self.crop_x, x)

        if m.ndim == 3:
            result = m[:, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
            return np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect')
        else:
            channels = []
            for c in range(m.shape[0]):
                result = m[c][:, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
                channels.append(np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect'))
            return np.stack(channels, axis=0)


class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)  # Z
    ]

    def __init__(self, ignore_index=None, aggregate_affinities=False, append_label=False, **kwargs):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        :param aggregate_affinities: aggregate affinities with the same offset across Z,Y,X axes
        :param append_label: if True append the orignal ground truth labels to the last channel
        :param blur: Gaussian blur the boundaries
        :param sigma: standard deviation for Gaussian kernel
        """
        self.ignore_index = ignore_index
        self.aggregate_affinities = aggregate_affinities
        self.append_label = append_label

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()
        boundary_arr = [np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels]
        channels = np.stack(boundary_arr)
        results = []
        if self.aggregate_affinities:
            assert len(kernels) % 3 == 0, "Number of kernels must be divided by 3 (one kernel per offset per Z,Y,X axes"
            # aggregate affinities with the same offset
            for i in range(0, len(kernels), 3):
                # merge across X,Y,Z axes (logical OR)
                xyz_aggregated_affinities = np.logical_or.reduce(channels[i:i + 3, ...]).astype(np.int32)
                # recover ignore index
                xyz_aggregated_affinities = _recover_ignore_index(xyz_aggregated_affinities, m, self.ignore_index)
                results.append(xyz_aggregated_affinities)
        else:
            results = [_recover_ignore_index(channels[i], m, self.ignore_index) for i in range(channels.shape[0])]

        if self.append_label:
            # append original input data
            results.append(m)

        # stack across channel dim
        return np.stack(results, axis=0)

    @staticmethod
    def create_kernel(axis, offset):
        """
        Create a 3D convolution kernel for an integer offset along a given axis
        and return it transposed according to 'axis'.

        The original code assumed offset was positive, creating an array of shape
        (1,1, offset+1). Now we handle negative offsets as well by using abs(offset).

        If offset > 0:
            - place +1 at index=0
            - place -1 at index=offset
        If offset < 0:
            - place +1 at index=abs(offset)
            - place -1 at index=0

        Then we transpose it so that the dimension of size `abs(offset)+1` goes
        into the specified axis position, matching the original logic.

        :param axis: A tuple like (0,1,2) or (2,0,1) indicating how to transpose
                     the kernel to place the 'offset' dimension along X, Y, or Z.
        :param offset: Integer offset, can be positive or negative.
        :return: np.ndarray of shape (kD, kH, kW) with +1 and -1 placed accordingly.
        """
        abs_off = abs(offset)
        k_size = abs_off + 1  # total size along the 'offset' dimension

        # Create a minimal kernel of shape (1,1,k_size),
        # which we'll then transpose to the correct axis.
        k = np.zeros((1, 1, k_size), dtype=np.int32)

        if offset >= 0:
            # offset >= 0 => put +1 at index=0, -1 at index=offset
            # e.g. offset=3 => shape=(1,1,4); k[0,0,0]=+1, k[0,0,3]=-1
            k[0, 0, 0] = 1
            k[0, 0, offset] = -1
        else:
            # offset < 0 => put +1 at index=abs_off, -1 at index=0
            # e.g. offset=-2 => shape=(1,1,3); k[0,0,2]=+1, k[0,0,0]=-1
            k[0, 0, abs_off] = 1
            k[0, 0, 0] = -1

        # Now transpose so that the dimension of size k_size
        # goes to the axis indicated by 'axis'
        #
        # Example: if axis=(2,0,1), we want the dimension of size k_size
        # to end up in dimension 2 (the 'Z' dimension).
        # np.transpose(k, axis) reorders k's axes from (0,1,2) to (axis[0], axis[1], axis[2]).
        return np.transpose(k, axis)


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, mode='thick', foreground=False,
                 **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype('int32')

        results = []
        if self.foreground:
            foreground = (m > 0).astype('uint8')
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class BlobsToMask:
    """
    Returns binary mask from labeled image, i.e. every label greater than 0 is treated as foreground.

    """

    def __init__(self, append_label=False, boundary=False, cross_entropy=False, **kwargs):
        self.cross_entropy = cross_entropy
        self.boundary = boundary
        self.append_label = append_label

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        mask = (m > 0).astype('uint8')
        results = [mask]

        if self.boundary:
            outer = find_boundaries(m, connectivity=2, mode='outer')
            if self.cross_entropy:
                # boundary is class 2
                mask[outer > 0] = 2
                results = [mask]
            else:
                results.append(outer)

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class RandomLabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border. Then the offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the ground
    truth  (think of it as a boundary denoising scheme).
    """

    def __init__(self, random_state, max_offset=10, ignore_index=None, append_label=False, z_offset_scale=2, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label, aggregate_affinities=False)
        self.random_state = random_state
        self.offsets = tuple(range(1, max_offset + 1))
        self.z_offset_scale = z_offset_scale

    def get_kernels(self):
        rand_offset = self.random_state.choice(self.offsets)
        axis_ind = self.random_state.randint(3)
        # scale down z-affinities due to anisotropy
        if axis_ind == 2:
            rand_offset = max(1, rand_offset // self.z_offset_scale)

        rand_axis = self.AXES_TRANSPOSE[axis_ind]
        # return a single kernel
        return [self.create_kernel(rand_axis, rand_offset)]


from cc3d import connected_components  # Using connected-components-3d library for better performance

class LabelToAffinities(AbstractLabelToBoundary):
    """
    First converts binary mask to instance labels using connected components,
    then converts to affinity maps using the instance labels.
    Generates both + and - offsets for x, y, z in the exact order:
        0 => +x,  1 => -x,
        2 => +y,  3 => -y,
        4 => +z,  5 => -z
    for each offset in offsets / z_offsets.
    """

    # We’ll keep AXES_TRANSPOSE for reference, but won’t rely on it
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)   # Z
    ]

    def __init__(self, offsets, ignore_index=None, append_label=False,
                 aggregate_affinities=False, z_offsets=None, **kwargs):

        super().__init__(ignore_index=ignore_index,
                         append_label=append_label,
                         aggregate_affinities=aggregate_affinities)

        assert isinstance(offsets, (list, tuple)), "'offsets' must be a list or tuple"
        assert all(a > 0 for a in offsets), "'offsets' must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"

        # If z_offsets not provided, reuse the xy offsets
        if z_offsets is not None:
            assert len(offsets) == len(z_offsets), (
                "z_offsets length must match offsets length"
            )
        else:
            z_offsets = list(offsets)

        self.z_offsets = z_offsets

        # Build kernels in the order:
        #   +x, -x, +y, -y, +z, -z
        # for each offset in offsets[] and z_offsets[].
        self.kernels = []
        for xy_offset, z_offset in zip(offsets, z_offsets):
            # +x
            self.kernels.append(self.create_kernel((0,1,2), xy_offset))
            # -x
            self.kernels.append(self.create_kernel((0,1,2), -xy_offset))

            # +y
            self.kernels.append(self.create_kernel((0,2,1), xy_offset))
            # -y
            self.kernels.append(self.create_kernel((0,2,1), -xy_offset))

            # +z
            self.kernels.append(self.create_kernel((2,0,1), z_offset))
            # -z
            self.kernels.append(self.create_kernel((2,0,1), -z_offset))

    def __call__(self, m):
        """
        Override the parent call method to first create instance labels
        """
        assert m.ndim == 3, "Input must be 3D"

        # Convert binary mask to instance labels using connected components
        binary_mask = (m > 0).astype(np.uint32)
        instance_labels = connected_components(binary_mask)

        # Now process with parent class using instance labels
        return super().__call__(instance_labels)

    def get_kernels(self):
        return self.kernels


class LabelToZAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels (which can be seen
    as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf)
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), 'offsets must be a list or a tuple'
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"

        self.kernels = []
        z_axis = self.AXES_TRANSPOSE[2]
        # create kernels
        for z_offset in offsets:
            self.kernels.append(self.create_kernel(z_axis, z_offset))

    def get_kernels(self):
        return self.kernels


class LabelToBoundaryAndAffinities:
    """
    Combines the StandardLabelToBoundary and LabelToAffinities in the hope
    that that training the network to predict both would improve the main task: boundary prediction.
    """

    def __init__(self, xy_offsets, z_offsets, append_label=False, blur=False, sigma=1, ignore_index=None, mode='thick',
                 foreground=False, **kwargs):
        # blur only StandardLabelToBoundary results; we don't want to blur the affinities
        self.l2b = StandardLabelToBoundary(blur=blur, sigma=sigma, ignore_index=ignore_index, mode=mode,
                                           foreground=foreground)
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        boundary = self.l2b(m)
        affinities = self.l2a(m)
        return np.concatenate((boundary, affinities), axis=0)


class LabelToMaskAndAffinities:
    def __init__(self, xy_offsets, z_offsets, append_label=False, background=0, ignore_index=None, **kwargs):
        self.background = background
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        mask = m > self.background
        mask = np.expand_dims(mask.astype(np.uint8), axis=0)
        affinities = self.l2a(m)
        return np.concatenate((mask, affinities), axis=0)


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class PercentileNormalizer:
    def __init__(self, pmin=1, pmax=99.6, channelwise=False, eps=1e-10, **kwargs):
        self.eps = eps
        self.pmin = pmin
        self.pmax = pmax
        self.channelwise = channelwise

    def __call__(self, m):
        if self.channelwise:
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            pmin = np.percentile(m, self.pmin, axis=axes, keepdims=True)
            pmax = np.percentile(m, self.pmax, axis=axes, keepdims=True)
        else:
            pmin = np.percentile(m, self.pmin)
            pmax = np.percentile(m, self.pmax)

        return (m - pmin) / (pmax - pmin + self.eps)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data
    in a fixed range of [-1, 1] or in case of norm01==True to [0, 1]. In addition, data can be
    clipped by specifying min_value/max_value either globally using single values or via a
    list/tuple channelwise if enabled.
    """

    def __init__(self, min_value=None, max_value=None, norm01=False, channelwise=False,
                 eps=1e-10, **kwargs):
        if min_value is not None and max_value is not None:
            assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.norm01 = norm01
        self.channelwise = channelwise
        self.eps = eps

    def __call__(self, m):
        if self.channelwise:
            # get min/max channelwise
            axes = list(range(m.ndim))
            axes = tuple(axes[1:])
            if self.min_value is None or 'None' in self.min_value:
                min_value = np.min(m, axis=axes, keepdims=True)

            if self.max_value is None or 'None' in self.max_value:
                max_value = np.max(m, axis=axes, keepdims=True)

            # check if non None in self.min_value/self.max_value
            # if present and if so copy value to min_value
            if self.min_value is not None:
                for i,v in enumerate(self.min_value):
                    if v != 'None':
                        min_value[i] = v

            if self.max_value is not None:
                for i,v in enumerate(self.max_value):
                    if v != 'None':
                        max_value[i] = v
        else:
            if self.min_value is None:
                min_value = np.min(m)
            else:
                min_value = self.min_value

            if self.max_value is None:
                max_value = np.max(m)
            else:
                max_value = self.max_value

        # calculate norm_0_1 with min_value / max_value with the same dimension
        # in case of channelwise application
        norm_0_1 = (m - min_value) / (max_value - min_value + self.eps)

        if self.norm01 is True:
          return np.clip(norm_0_1, 0, 1)
        else:
          return np.clip(2 * norm_0_1 - 1, -1, 1)


class AdditiveGaussianNoise:
    def __init__(self, random_state, scale=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise
        return m


class AdditivePoissonNoise:
    def __init__(self, random_state, lam=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m


class RicianNoiseTransform:
    """
    Adds Rician noise with given variance. MRI data typically has Rician distribution noise.
    """

    def __init__(self, random_state, noise_variance=(0, 0.1), execution_probability=0.5):
        self.random_state = random_state
        self.noise_variance = noise_variance if isinstance(noise_variance, tuple) else (0, noise_variance)
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            variance = self.random_state.uniform(*self.noise_variance)
            if isinstance(m, np.ndarray):
                noise1 = self.random_state.normal(0, np.sqrt(variance), size=m.shape)
                noise2 = self.random_state.normal(0, np.sqrt(variance), size=m.shape)
                return np.sqrt((m + noise1) ** 2 + noise2 ** 2)
            else:
                device = m.device
                noise1 = torch.normal(0, np.sqrt(variance), size=m.shape, device=device)
                noise2 = torch.normal(0, np.sqrt(variance), size=m.shape, device=device)
                return torch.sqrt((m + noise1) ** 2 + noise2 ** 2)
        return m

class BlankRectangleTransform:
    def __init__(self, random_state,
                 min_size=(10, 10, 10),
                 max_size=(30, 30, 30),
                 num_rectangles=(1, 3),
                 value_range=(0.0, 0.2),
                 execution_probability=0.5):
        self.random_state = random_state
        self.min_size = np.array(min_size)
        self.max_size = np.array(max_size)
        self.num_rectangles = num_rectangles if isinstance(num_rectangles, tuple) else (num_rectangles, num_rectangles + 1)
        self.value_range = value_range
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            if isinstance(m, np.ndarray):
                result = m.copy()
            else:
                result = m.clone()

            # Number of rectangles to add
            n_rect = self.random_state.randint(self.num_rectangles[0], self.num_rectangles[1])

            # Decide how to interpret shape
            # If m.ndim=4 => (C, D, H, W); we want (D, H, W) for indexing
            # If m.ndim=3 => (D, H, W)
            if m.ndim == 4:
                shape_for_op = m.shape[1:]   # skip channel dimension
            else:
                shape_for_op = m.shape       # use entire shape

            size = self.random_state.randint(self.min_size, self.max_size + 1)

            for _ in range(n_rect):
                # For 3D patching => shape_for_op has length 3 => (D,H,W)
                # shape_for_op[i] - size[i]
                pos = [self.random_state.randint(0, max(shape_for_op[i] - size[i], 1))
                       for i in range(len(size))]

                # Random intensity value
                value = self.random_state.uniform(*self.value_range)

                # Construct slices
                if m.ndim == 4:
                    # first dimension is channels => slice(None)
                    slices = tuple([slice(None)] +
                                   [slice(p, p + s) for p, s in zip(pos, size)])
                else:
                    # No channel dimension
                    slices = tuple([slice(p, p + s) for p, s in zip(pos, size)])

                # Apply rectangle
                if isinstance(m, np.ndarray):
                    result[slices] = value
                else:
                    import torch
                    result[slices] = torch.tensor(value,
                                                  device=result.device,
                                                  dtype=result.dtype)

            return result
        return m



class ContrastTransform:
    """
    Adjusts image contrast by scaling around mean value.
    """

    def __init__(self, random_state,
                 contrast_range=(0.75, 1.25),
                 preserve_range=True,
                 synchronize_channels=True,
                 p_per_channel=1.0,
                 execution_probability=0.5):
        self.random_state = random_state
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            if isinstance(m, np.ndarray):
                result = m.copy()
            else:
                result = m.clone()

            # Determine which channels to apply to
            num_channels = result.shape[0]
            apply_to_channel = [c for c in range(num_channels)
                                if self.random_state.uniform() < self.p_per_channel]

            if not apply_to_channel:  # If no channels selected
                return result

            # Generate contrast factors
            if self.synchronize_channels:
                # Same contrast for all channels
                multiplier = self.random_state.uniform(*self.contrast_range)
                multipliers = [multiplier] * len(apply_to_channel)
            else:
                # Different contrast for each channel
                multipliers = [self.random_state.uniform(*self.contrast_range)
                               for _ in apply_to_channel]

            # Apply contrast adjustment per channel
            for c, multiplier in zip(apply_to_channel, multipliers):
                # Store original range if needed
                if self.preserve_range:
                    if isinstance(m, np.ndarray):
                        minm = np.min(result[c])
                        maxm = np.max(result[c])
                    else:
                        minm = torch.min(result[c])
                        maxm = torch.max(result[c])

                # Compute mean
                if isinstance(m, np.ndarray):
                    mean = np.mean(result[c])
                    # Apply contrast adjustment
                    result[c] = (result[c] - mean) * multiplier + mean
                    # Preserve range if requested
                    if self.preserve_range:
                        result[c] = np.clip(result[c], minm, maxm)
                else:
                    mean = torch.mean(result[c])
                    # Apply contrast adjustment
                    result[c] = (result[c] - mean) * multiplier + mean
                    # Preserve range if requested
                    if self.preserve_range:
                        result[c].clamp_(minm, maxm)

            return result
        return m

class GammaTransform:
    """
    Applies gamma correction to images.
    """

    def __init__(self, random_state,
                 gamma_range=(0.7, 1.5),
                 invert_image_prob=0.0,
                 retain_stats_prob=1.0,
                 execution_probability=0.5):
        self.random_state = random_state
        self.gamma_range = gamma_range
        self.invert_image_prob = invert_image_prob
        self.retain_stats_prob = retain_stats_prob
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            if isinstance(m, np.ndarray):
                result = m.copy()
            else:
                result = m.clone()
                result = result.cpu().numpy()

            # Per-channel operations
            for c in range(result.shape[0]):
                # Check if we should invert
                if self.random_state.uniform() < self.invert_image_prob:
                    result[c] *= -1

                # Store stats if we need to retain them
                retain_stats = self.random_state.uniform() < self.retain_stats_prob
                if retain_stats:
                    mean = np.mean(result[c])
                    std = np.std(result[c])

                # Apply gamma correction
                minm = np.min(result[c])
                rnge = np.max(result[c]) - minm
                gamma = self.random_state.uniform(*self.gamma_range)

                # Avoid division by zero
                rnge = np.maximum(rnge, 1e-7)
                result[c] = np.power((result[c] - minm) / rnge, gamma) * rnge + minm

                # Restore stats if needed
                if retain_stats:
                    curr_mean = np.mean(result[c])
                    curr_std = np.std(result[c])
                    if curr_std > 1e-7:  # Avoid division by zero
                        result[c] = (result[c] - curr_mean) * (std / curr_std) + mean

                # Restore inversion if needed
                if self.random_state.uniform() < self.invert_image_prob:
                    result[c] *= -1

            return torch.from_numpy(result) if isinstance(m, torch.Tensor) else result
        return m

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Relabel:
    """
    Relabel a numpy array of labels into a consecutive numbers, e.g.
    [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, append_original=False, run_cc=True, ignore_label=None, **kwargs):
        self.append_original = append_original
        self.ignore_label = ignore_label
        self.run_cc = run_cc

        if ignore_label is not None:
            assert append_original, "ignore_label present, so append_original must be true, so that one can localize the ignore region"

    def __call__(self, m):
        orig = m
        if self.run_cc:
            # assign 0 to the ignore region
            m = measure.label(m, background=self.ignore_label)

        _, unique_labels = np.unique(m, return_inverse=True)
        result = unique_labels.reshape(m.shape)
        if self.append_original:
            result = np.stack([result, orig])
        return result


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m


class RgbToLabel:
    def __call__(self, img):
        img = np.array(img)
        assert img.ndim == 3 and img.shape[2] == 3
        result = img[..., 0] * 65536 + img[..., 1] * 256 + img[..., 2]
        return result


class LabelToTensor:
    def __call__(self, m):
        m = np.array(m)
        return torch.from_numpy(m.astype(dtype='int64'))


class GaussianBlur3D:
    def __init__(self, sigma=[.1, 2.], execution_probability=0.5, **kwargs):
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, x):
        if random.random() < self.execution_probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = gaussian(x, sigma=sigma)
            return x
        return x


class Transformer:
    def __init__(self, phase_config, base_config):
        self.phase_config = phase_config
        self.config_base = base_config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('pytorch3dunet.augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
