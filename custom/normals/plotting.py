import cv2
import imageio
import numpy as np
import torch
from pathlib import Path

def save_debug_gif(input_volume, sheet_truth, normal_truth, affinity_truth, outputs, save_prefix="evals/img"):
    with torch.no_grad():
        # Process model outputs
        sheet_pred = torch.sigmoid(outputs['sheet']).squeeze().cpu().numpy()
        normal_pred = outputs['normals'].squeeze().cpu().numpy()
        affinity_pred = torch.sigmoid(outputs['affinities']).squeeze().cpu().numpy()

        # Process ground truth
        input_volume = input_volume.squeeze().cpu().numpy()
        sheet_truth = sheet_truth.squeeze().cpu().numpy()
        normal_truth = normal_truth.squeeze().cpu().numpy()
        affinity_truth = affinity_truth.squeeze().cpu().numpy()

        # Normalize normal vectors
        normal_pred /= np.linalg.norm(normal_pred, axis=0, keepdims=True)
        normal_pred[:, np.isnan(normal_pred).all(axis=0)] = 0

        normal_truth /= np.linalg.norm(normal_truth, axis=0, keepdims=True)
        normal_truth[:, np.isnan(normal_truth).all(axis=0)] = 0

        imgs = []
        for k in range(input_volume.shape[-1]):  # Loop through the depth dimension
            # Input volume slice
            slice_volume = cv2.cvtColor((input_volume[..., k] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Ground truth slices
            slice_sheet_truth = cv2.cvtColor((sheet_truth[..., k] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            slice_normals_truth = ((normal_truth[::-1, ..., k] * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)
            # Take first 3 channels of affinity for visualization
            slice_affinity_truth = cv2.cvtColor((affinity_truth[0, ..., k] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Prediction slices
            slice_sheet_pred = cv2.cvtColor((sheet_pred[..., k] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            slice_normals_pred = ((normal_pred[::-1, ..., k] * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)
            slice_affinity_pred = cv2.cvtColor((affinity_pred[0, ..., k] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Normal vector magnitude
            slice_normals_abs = cv2.cvtColor(
                ((np.linalg.norm(slice_normals_pred, axis=-1) /
                  np.linalg.norm(slice_normals_pred, axis=-1).max()) * 255).astype(np.uint8),
                cv2.COLOR_GRAY2BGR
            )

            # Stack images
            img = np.vstack([
                np.hstack([slice_volume, slice_sheet_truth, slice_normals_truth, slice_affinity_truth]),
                np.hstack([slice_normals_abs, slice_sheet_pred, slice_normals_pred, slice_affinity_pred])
            ])
            imgs.append(img)

        # Save gif
        save_path = Path(save_prefix).parent
        save_path.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(f"{save_prefix}.gif", imgs, fps=24)