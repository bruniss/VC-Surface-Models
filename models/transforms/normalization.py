import numpy as np

def ct_normalize(volume, clip_min, clip_max, global_mean, global_std):
    """
    Mimics nnU-Net CT normalization:
      1) Clip to [clip_min, clip_max],
      2) Subtract mean,
      3) Divide by std.
    """
    volume = volume.astype(np.float32)
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - global_mean) / global_std
    return volume