import torch
import torch.nn.functional as F

def normal_cosine_loss(pred, target):
    """
    pred:   [B, 3, D, H, W] predicted normals
    target: [B, 3, D, H, W] ground-truth normals
    Returns a scalar tensor = 1 - mean_cosine_similarity
    """
    # Compute cosine similarity over the channel dimension (dim=1),
    # ignoring batch (B) and spatial (D,H,W) dims.
    # eps avoids divide-by-zero if vectors are zero-length
    cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-8)
    return 1.0 - cos_sim.mean()
