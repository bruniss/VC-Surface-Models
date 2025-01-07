import cv2
import imageio
import numpy as np
import torch
from pathlib import Path

def save_debug_gif(input_volume, sheet_truth, normal_truth, outputs, save_prefix="evals/img"):
    """
    Creates a GIF with 8 panels (2 rows x 4 columns) per slice Z:
      Top row:    [ input, sheet_truth, normals_truth, affinity_truth ]
      Bottom row: [ normal_magnitude, sheet_pred, normals_pred, affinity_pred ]
    Each panel is a 2D slice at depth z, so you see XY planes for each z in range(Z).
    """
    with torch.no_grad():
        # ---------------------------------------------------------------------
        # 1) Move all model outputs & ground truths to numpy arrays
        #    By the time you call squeeze(), shapes should be:
        #      * input_volume, sheet_truth => (Z, Y, X)
        #      * normal_truth, normal_pred => (3, Z, Y, X)
        #      * affinity_truth, affinity_pred => (C, Z, Y, X)
        # ---------------------------------------------------------------------
        sheet_pred = torch.sigmoid(outputs['sheet']).squeeze().cpu().numpy()       # (Z, Y, X)
        normal_pred = outputs['normals'].squeeze().cpu().numpy()                  # (3, Z, Y, X)
        # affinity_pred = torch.sigmoid(outputs['affinities']).squeeze().cpu().numpy()  # (C, Z, Y, X)

        input_volume = input_volume.squeeze().cpu().numpy()   # (Z, Y, X)
        sheet_truth = sheet_truth.squeeze().cpu().numpy()     # (Z, Y, X)
        normal_truth = normal_truth.squeeze().cpu().numpy()   # (3, Z, Y, X)
        # affinity_truth = affinity_truth.squeeze().cpu().numpy()   # (C, Z, Y, X)

        # ---------------------------------------------------------------------
        # 2) (Optional) Normalize normal vectors to unit-length if desired
        #    This is per-voxel normalization. Keep in mind axis=0 => channel
        #    so normal_pred[:, z, y, x] is the 3D vector at (z,y,x).
        #    If you do this, be sure to handle zero or NaN norms carefully.
        # ---------------------------------------------------------------------
        # Example approach (comment out if you prefer not to do this step):
        def safe_normalize(vectors):
            # vectors shape = (3, Z, Y, X)
            norms = np.linalg.norm(vectors, axis=0, keepdims=True)  # shape (1, Z, Y, X)
            # avoid divide-by-zero
            norms = np.where(norms == 0, 1.0, norms)
            return vectors / norms

        normal_pred = safe_normalize(normal_pred)
        normal_truth = safe_normalize(normal_truth)

        # ---------------------------------------------------------------------
        # 3) Prepare to build our GIF frames
        # ---------------------------------------------------------------------
        imgs = []
        num_slices = input_volume.shape[0]  # = Z

        for z in range(num_slices):
            # ~~~~~~~~~ TOP ROW ~~~~~~~~~
            # (A) Input volume slice => shape (Y, X)
            slice_input = input_volume[z, ...]
            slice_input_8u = (slice_input * 255).clip(0, 255).astype(np.uint8)
            slice_input_bgr = cv2.cvtColor(slice_input_8u, cv2.COLOR_GRAY2BGR)

            # (B) Sheet truth => shape (Y, X)
            slice_sheet_truth = (sheet_truth[z, ...] * 255).clip(0, 255).astype(np.uint8)
            slice_sheet_truth_bgr = cv2.cvtColor(slice_sheet_truth, cv2.COLOR_GRAY2BGR)

            # (C) Normal truth => shape (3, Y, X)
            nt = normal_truth[:, z, ...]         # (3, Y, X)
            # map [-1..1] => [0..1] => [0..255]
            nt_mapped = ((nt * 0.5) + 0.5) * 255
            nt_mapped = nt_mapped.clip(0, 255).astype(np.uint8)
            slice_normals_truth = np.transpose(nt_mapped, (1, 2, 0))  # => (Y, X, 3) for display

            # (D) Affinity truth => e.g. show channel 0 => shape (Y, X)
            #slice_aff_truth = (affinity_truth[0, z, ...] * 255).clip(0, 255).astype(np.uint8)
            #slice_aff_truth_bgr = cv2.cvtColor(slice_aff_truth, cv2.COLOR_GRAY2BGR)

            top_row = np.hstack([
                slice_input_bgr,
                slice_sheet_truth_bgr,
                slice_normals_truth
                #slice_aff_truth_bgr
            ])

            # ~~~~~~~~~ BOTTOM ROW ~~~~~~~~~
            # (E) Normal magnitude => use predicted normal
            # shape (3, Y, X) => after mapping => shape (Y, X, 3)
            npred = normal_pred[:, z, ...]
            npred_mapped = ((npred * 0.5) + 0.5) * 255
            npred_mapped = npred_mapped.clip(0, 255).astype(np.uint8)
            slice_normals_pred_bgr = np.transpose(npred_mapped, (1, 2, 0))  # (Y, X, 3)

            # Magnitude in grayscale for debugging
            mag = np.linalg.norm(slice_normals_pred_bgr, axis=-1)  # shape (Y, X)
            maxval = mag.max() if mag.max() > 0 else 1
            mag_norm = (mag / maxval * 255).astype(np.uint8)
            slice_normals_abs_bgr = cv2.cvtColor(mag_norm, cv2.COLOR_GRAY2BGR)

            # (F) Sheet prediction => shape (Z, Y, X)
            slice_sheet_pred = (sheet_pred[z, ...] * 255).clip(0, 255).astype(np.uint8)
            slice_sheet_pred_bgr = cv2.cvtColor(slice_sheet_pred, cv2.COLOR_GRAY2BGR)

            # (G) Affinity prediction => e.g. channel 0
            #slice_aff_pred = (affinity_pred[0, z, ...] * 255).clip(0, 255).astype(np.uint8)
            #slice_aff_pred_bgr = cv2.cvtColor(slice_aff_pred, cv2.COLOR_GRAY2BGR)

            bottom_row = np.hstack([
                slice_normals_abs_bgr,
                slice_sheet_pred_bgr,
                slice_normals_pred_bgr
                #slice_aff_pred_bgr
            ])

            # Stack rows
            final_img = np.vstack([top_row, bottom_row])
            imgs.append(final_img)

        # ---------------------------------------------------------------------
        # 4) Save the animation as a GIF
        # ---------------------------------------------------------------------
        save_path = Path(save_prefix).parent
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving GIF to {save_prefix}.gif")
        imageio.mimsave(f"{save_prefix}.gif", imgs, fps=24)
