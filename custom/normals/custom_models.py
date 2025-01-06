from pytorch3dunet.unet3d.model import ResidualUNetSE3D, MultiTaskResidualUNetSE3D
from pytorch3dunet.unet3d.buildingblocks import create_decoders, ResNetBlockSE
from pytorch3dunet.augment.transforms import (
    Compose, LabelToAffinities, Standardize,
    RandomFlip, RandomRotate90, GaussianBlur3D, BlankRectangleTransform, RicianNoiseTransform,
    ContrastTransform, GammaTransform)
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from dataset import ZarrSegmentationDataset3D
from pathlib import Path
from plotting import save_debug_gif
from monai.losses.dice import DiceCELoss


def train():
    model = MultiTaskResidualUNetSE3D(
        in_channels=1,
        sheet_channels=1,
        normal_channels=3,
        xy_offsets=[1, 3],  # affinity map offsets, each makes 3 channels for xyz, this explodes very fast be careful
        z_offsets=[1, 3],
        f_maps=[32, 64, 128, 256, 320, 320, 320],
        num_levels=7
    )


    random_state = np.random.RandomState(47)

    transforms_list = [
        (RandomFlip(random_state=random_state, execution_probability=0.3),
         ['image', 'sheet_label', 'normals', 'affinity']),
        (RandomRotate90(random_state=random_state, execution_probability=0.3),
         ['image', 'sheet_label', 'normals', 'affinity']),
        (GaussianBlur3D(sigma=[0.1, 2.0], execution_probability=0.2),
         ['image']),
        (ContrastTransform(execution_probability=0.2, random_state=random_state),
         ['image']),
        (GammaTransform(execution_probability=0.2, random_state=random_state),
         ['image']),
        (BlankRectangleTransform(execution_probability=0.2, random_state=random_state),
         ['image', 'normals']),
        (RicianNoiseTransform(execution_probability=0.2, random_state=random_state),
         ['image']),

    ]

    dataset = ZarrSegmentationDataset3D(
        volume_path=Path('/path/to/volume.zarr'),
        sheet_label_path=Path('/mnt/raid_nvme/datasets/1-voxel-sheet_slices-closed.zarr/0.zarr'),
        normals_path=Path('/home/sean/Documents/GitHub/VC-Surface-Models/custom/normals.zarr'),
        patch_size=(192, 192, 192),
        min_labeled_ratio=0.1,
        xy_offsets=[1, 3],
        z_offsets=[1, 3],
        transforms_list=transforms_list,

    )

    device = torch.device('cuda')
    model = model.to(device)
    model_name = 'SheetNormAffinity'

    # --- losses ---- #
    sheet_criterion = DiceCELoss(label_smoothing=0.1, jaccard=True) # dice and cross-entropy for sheet
    normal_criterion = nn.MSELoss() # mean squared error for normals (might not be best choice idk yet)
    affinity_criterion = nn.BCEWithLogitsLoss() # i want to experiment with this, but for now this will work
    w_sheet, w_normal, w_affinity = 1.0, 1.0, 1.0 # equally weighting all for now

    # ---- optimizer seutp, scaler setup, splits and batch setup ---- #
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.7,
                                  patience=30,
                                  min_lr = 1e-6,
                                  verbose=True)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_split = 0.8
    split = int(np.floor(train_split * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    batch_size = 2

    # apply gradient accumulation
    grad_accumulate_n = 6

    scaler = torch.cuda.amp.GradScaler()


    # ---- dataloader ----- #
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                            pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(val_indices),
                                pin_memory=True, num_workers=1)


    # ---- training! ----- #
    for epoch in range(0, 500):
        model.train()
        running_loss = 0.0
        running_loss_sheet = 0.0
        running_loss_normal = 0.0
        running_loss_affinity = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in pbar:
            inputs, sheet_labels, normal_labels, affinity_labels = data
            inputs = inputs.to(device)
            sheet_labels = sheet_labels.to(device).float() # i really
            normal_labels = normal_labels.to(device).float() # need to decide
            affinity_labels = affinity_labels.to(device).float() # if this level of precision is even needed

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                sheet_loss = sheet_criterion(outputs['sheet'], sheet_labels)
                normal_loss = normal_criterion(outputs['normals'], normal_labels)
                affinity_loss = affinity_criterion(outputs['affinities'], affinity_labels)
                loss = w_sheet * sheet_loss + w_normal * normal_loss + w_affinity * affinity_loss

            running_loss += loss.item()
            running_loss_sheet += sheet_loss.item()
            running_loss_normal += normal_loss.item()
            running_loss_affinity += affinity_loss.item()

            scaler.scale(loss).backward()
            if (i + 1) % grad_accumulate_n == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

            avg_sheet_loss = running_loss_sheet / ((i + 1) / grad_accumulate_n)
            avg_normal_loss = running_loss_normal / ((i + 1) / grad_accumulate_n)
            avg_affinity_loss = running_loss_affinity / ((i + 1) / grad_accumulate_n)
            avg_running_loss = running_loss / ((i + 1) / grad_accumulate_n)

            pbar.set_description(f"Epoch: {epoch + 1} | Loss: {avg_running_loss:.4f} | "
                                 f"Sheet Loss: {avg_sheet_loss:.4f} | Normal Loss: {avg_normal_loss:.4f} | "
                                 f"Affinity Loss: {avg_affinity_loss:.4f}")
        pbar.close()

        epoch_avg_sheet_loss = running_loss_sheet / len(dataloader)
        epoch_avg_normal_loss = running_loss_normal / len(dataloader)
        epoch_avg_affinity_loss = running_loss_affinity / len(dataloader)
        epoch_avg_loss = running_loss / (len(dataloader))

        print(f"Epoch: {epoch + 1} | Sheet Loss: {epoch_avg_sheet_loss:.4f} | "
              f"Normal Loss: {epoch_avg_normal_loss:.4f} | Affinity Loss: {epoch_avg_affinity_loss:.4f}")

        print(
            f'\End of Epoch {epoch + 1} | Average Sheet Loss: {epoch_avg_sheet_loss:.4f} | Average Normal Loss: {epoch_avg_normal_loss:.4f} | Average Affinity Loss: {epoch_avg_affinity_loss:.4f} | Average Loss: {epoch_avg_loss}\n'
        )
        torch.save(model.state_dict(), f'{model_name}_{epoch + 1}.pth')
        torch.save(optimizer.state_dict(), f'{model_name}_{epoch + 1}_optimizer.pth')
        torch.save(scheduler.state_dict(), f'{model_name}_{epoch + 1}_scheduler.pth')

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                val_running_loss_sheet = 0.0
                val_running_loss_normal = 0.0
                val_running_loss_affinity = 0.0
                pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
                for i, data in pbar:
                    inputs, sheet_labels, normal_labels, affinity_labels = data
                    inputs = inputs.to(device)
                    sheet_labels = sheet_labels.to(device).float()
                    normal_labels = normal_labels.to(device).float()
                    affinity_labels = affinity_labels.to(device).float()

                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        sheet_loss = sheet_criterion(outputs['sheet'], sheet_labels)
                        normal_loss = normal_criterion(outputs['normals'], normal_labels)
                        affinity_loss = affinity_criterion(outputs['affinities'], affinity_labels)
                        loss = w_sheet * sheet_loss + w_normal * normal_loss + w_affinity * affinity_loss

                        if i == 0:
                            save_debug_gif(inputs, sheet_labels, normal_labels, affinity_labels, outputs, save_prefix=f"evals/img_e{epoch}")

                    val_running_loss += loss.item()
                    val_running_loss_sheet += sheet_loss.item()
                    val_running_loss_normal += normal_loss.item()
                    val_running_loss_affinity += affinity_loss.item()

                    avg_val_running_loss = val_running_loss / (i + 1)
                    avg_val_sheet_loss = val_running_loss_sheet / (i + 1)
                    avg_val_normal_loss = val_running_loss_normal / (i + 1)
                    avg_val_affinity_loss = val_running_loss_affinity / (i + 1)

                    # Update progress bar
                    pbar.set_description(
                        f'Val Running Loss: {avg_val_running_loss:.8f}, Sheet Loss: {avg_val_sheet_loss:.8f}, Normal Loss: {avg_val_normal_loss:.8f}, Affinity Loss: {avg_val_affinity_loss:.8f}')

            pbar.close()

            epoch_val_avg_sheet_loss = val_running_loss_sheet / len(val_dataloader)
            epoch_val_avg_normal_loss = val_running_loss_normal / len(val_dataloader)
            epoch_val_avg_affinity_loss = val_running_loss_affinity / len(val_dataloader)
            epoch_val_avg_loss = val_running_loss / len(val_dataloader)

            # Print average loss per batch after each epoch
            print(
                f'\nEnd of Epoch: {epoch + 1}, Val Average Sheet Loss: {epoch_val_avg_sheet_loss:.8f}, Val Average Normal Loss: {epoch_val_avg_normal_loss:.8f}, Val Average Affinity Loss: {epoch_val_avg_affinity_loss:.8f}, Val Average Loss: {epoch_val_avg_loss:.8f}\n')

            scheduler.step(epoch_val_avg_loss)

    print('Training Finished!')
    torch.save(model.state_dict(), f'{model_name}_final.pth')

if __name__ == '__main__':
    train()



# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]