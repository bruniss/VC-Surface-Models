import argparse
import os
import torch
import tifffile
import numpy as np

# --- import your model architecture ---
from pytorch3dunet.unet3d.model import MultiTaskResidualUNetSE3D

def run_inference(
    model_path: str,
    input_path: str,
    output_dir: str,
    in_channels: int = 1,
    sheet_channels: int = 1,
    normal_channels: int = 3,

):
    """
    Run inference on a 3D volume using a trained MultiTaskResidualUNetSE3D model.
    Args:
        model_path (str): Path to the .pth file containing trained weights.
        input_path (str): Path to the 3D image data (e.g., a TIFF stack).
        output_dir (str): Directory to save output predictions.
        in_channels (int): Number of input channels for the model (default 1).
        sheet_channels (int): Number of sheet channels in the output.
        normal_channels (int): Number of normal channels in the output.
        xy_offsets (list): Offsets for affinity maps in x/y (only if you used them in training).
        z_offsets (list): Offsets for affinity maps in z (only if you used them in training).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Instantiate the model with the same architecture as in training
    model = MultiTaskResidualUNetSE3D(
        in_channels=1,
        sheet_channels=1,
        normal_channels=3,
        f_maps=[32, 64, 128, 256, 320, 320, 320],
        num_levels=7
    )


    # 2) Load the trained weights
    print(f"[INFO] Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    # 3) Load the 3D volume (TIF stack, zarr, NIfTI, etc. - here we assume a TIF stack)
    print(f"[INFO] Loading input data from {input_path}")
    volume = tifffile.imread(input_path)  # shape: (D, H, W) or (Z, Y, X)

    # Ensure the volume has shape [1, 1, D, H, W] for the model:
    #   - 1 batch dimension
    #   - 1 channel (if in_channels=1)
    # If your data already has multiple channels, adjust accordingly.
    volume_torch = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)

    # 4) Run inference
    with torch.no_grad():
        outputs = model(volume_torch)  # dict: { 'sheet': [B,1,D,H,W], 'normals': [B,3,D,H,W], 'affinities': ... }

    # 5) Extract predictions from the output dict
    # sheet_pred shape:    [B, 1, D, H, W]
    # normals_pred shape:  [B, 3, D, H, W]
    # If you also trained affinities, you'd do outputs['affinities'] similarly.
    sheet_pred   = outputs['sheet'].cpu().numpy()      # -> [1, 1, D, H, W]
    normals_pred = outputs['normals'].cpu().numpy()    # -> [1, 3, D, H, W]

    # 6) Remove the batch dimension
    sheet_pred   = sheet_pred[0, 0]   # -> [D, H, W]
    normals_pred = normals_pred[0]    # -> [3, D, H, W]

    # 7) Save predictions to disk
    os.makedirs(output_dir, exist_ok=True)

    # Sheet segmentation output
    sheet_out_path = os.path.join(output_dir, "sheet_prediction.tif")
    tifffile.imwrite(sheet_out_path, sheet_pred.astype(np.float32))

    # Normal vectors output
    # Optionally rearrange from [3, D, H, W] -> [D, H, W, 3] for easier viewing in image tools
    normals_pred_zyxc = np.moveaxis(normals_pred, 0, -1)  # -> [D, H, W, 3]
    normals_out_path = os.path.join(output_dir, "normals_prediction.tif")
    tifffile.imwrite(normals_out_path, normals_pred_zyxc.astype(np.float32))

    print(f"[INFO] Inference complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for MultiTaskResidualUNetSE3D.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth weights")
    parser.add_argument("--input_path", type=str, required=True, help="Path to 3D input (tif, zarr, etc.)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")

    # If your model differs from the defaults, you can add more arguments here
    # e.g. in_channels, offsets, etc.

    args = parser.parse_args()

    # Run
    run_inference(
        model_path=args.model_path,
        input_path=args.input_path,
        output_dir=args.output_dir
    )
