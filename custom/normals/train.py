from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import AdamW
import numpy as np
from dataset import ZarrSegmentationDataset3D
from pathlib import Path
from plotting import save_debug_gif
from monai.losses.dice import DiceCELoss
import tifffile
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from losses import normal_cosine_loss, GeneralizedDiceLoss, masked_cosine_loss, BCEWithLogitsLossLabelSmoothing
from pytorch3dunet.unet3d.losses import BCEDiceLoss
import os


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
        #(RandomFlip(random_state=random_state, execution_probability=0.3),
         #['image', 'sheet_label', 'normals', 'affinity']),
        #(RandomRotate90(random_state=random_state, execution_probability=0.3),
         #['image', 'sheet_label', 'normals', 'affinity']),
        (GaussianBlur3D(sigma=[0.1, 2.0], execution_probability=0.2),
         ['image']),
        (ContrastTransform(execution_probability=0.2, random_state=random_state),
         ['image']),
        (GammaTransform(execution_probability=0.2, random_state=random_state),
         ['image']),
        (BlankRectangleTransform(execution_probability=0.2, random_state=random_state),
         ['image']),
        (RicianNoiseTransform(execution_probability=0.2, random_state=random_state),
         ['image']),

    ]

    dataset = ZarrSegmentationDataset3D(
        volume_path=Path('/mnt/raid_hdd/scrolls/s1/s1.zarr/0.zarr'),
        sheet_label_path=Path('/mnt/raid_nvme/datasets/1-voxel-sheet_slices-closed.zarr/0.zarr'),
        normals_path=Path('/home/sean/Documents/GitHub/VC-Surface-Models/custom/normals.zarr'),
        patch_size=(64, 192, 192),
        min_labeled_ratio=0.1,
        xy_offsets=None, #[1, 3],
        z_offsets=None, # 1, 3],
        transforms_list=transforms_list,
        use_cache=True,
        cache_file=Path('/home/sean/Documents/GitHub/VC-Surface-Models/custom/normals/valid_patch_cache64_192_192_boundary_fill.json')
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"volume_path: {dataset.volume_path}")
    print(f"sheet_label_path: {dataset.sheet_label_path}")
    print(f"normals_path: {dataset.normals_path}")
    print(f"patch_size: {dataset.patch_size}")

    device = torch.device('cuda')
    model = torch.compile(model)
    model = model.to(device)
    model_name = 'SheetNorm'


    # ---- optimizer seutp, scheduler setup, splits and batch setup ---- #
    max_steps_per_epoch = 500
    max_val_steps_per_epoch = 25
    max_epoch = 500
    initial_lr = 1e-3
    weight_decay = 1e-4
    use_affinities = False
    ckpt_out_base = '/mnt/raid_hdd/models/normals/checkpoints'
    os.makedirs(ckpt_out_base, exist_ok=True)
    checkpoint_path = None #"/mnt/raid_hdd/models/normals/checkpoints/SheetNormAffinity_50.pth"

    # --- losses ---- #
    sheet_criterion = BCEDiceLoss(alpha=0.5, beta=0.5)  # dice and cross-entropy for sheet
    # normal_criterion = nn.MSELoss() # mean squared error for normals (might not be best choice idk yet)
    normal_criterion = masked_cosine_loss
    # affinity_criterion = nn.BCEWithLogitsLoss() # i want to experiment with this, but for now this will work
    # affinity_criterion = GeneralizedDiceLoss(normalization='sigmoid')

    if use_affinities:
        affinity_criterion = BCEWithLogitsLossLabelSmoothing(smoothing=0.15, reduction='mean')
        w_sheet, w_normal, w_affinity = 1, 0.25, 1
    else:
        w_sheet, w_normal = 1, 0.5

    optimizer = AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr,
    #                            momentum=0.99,
    #                            #weight_decay=weight_decay,
    #                            nesterov=True)

    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=max_epoch,
                                  eta_min=0)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_split = 0.95
    split = int(np.floor(train_split * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    batch_size = 4

    # apply gradient accumulation
    grad_accumulate_n = 1

    scaler = torch.cuda.amp.GradScaler()

    # ---- dataloading ----- #

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                            pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(val_indices),
                                pin_memory=True, num_workers=4)

    start_epoch = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # If you saved them as a dictionary:
        model.load_state_dict(checkpoint) #['model']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = start_epoch
        print(f"Resuming training from epoch {start_epoch + 1}")

    # ---- training! ----- #
    for epoch in range(start_epoch, max_epoch):
        model.train()

        running_loss = 0.0
        running_loss_sheet = 0.0
        running_loss_normal = 0.0
        if use_affinities:
            running_loss_affinity = 0.0
        pbar = tqdm(enumerate(dataloader), total=max_steps_per_epoch)
        steps = 0

        for i, data in pbar:
            if i >= max_steps_per_epoch:
                break

            if use_affinities:
                inputs, sheet_labels, normal_labels, affinity_labels = data
            else:
                inputs, sheet_labels, normal_labels = data

            # print some info for the first batch
            if i == 0 and epoch == 0:
                print(f"inputs shape, type:       {inputs.shape} , {inputs.dtype}")
                print(f"sheet_labels shape, type: {sheet_labels.shape}, {sheet_labels.dtype}")
                print(f"sheet_label values: {torch.unique(sheet_labels)}")
                print(f"normal_labels shape, type:{normal_labels.shape}, {normal_labels.dtype}")
                if use_affinities:
                    print(f"affinity_labels shape, type:{affinity_labels.shape}, {affinity_labels.dtype}")

                batch_idx = 0
                # save input
                input_3d = inputs[batch_idx, 0].cpu().numpy()
                tifffile.imwrite("batch0_input.tif", input_3d)
                sheet_3d = sheet_labels[batch_idx, 0].cpu().numpy()
                tifffile.imwrite("batch0_sheet.tif", sheet_3d)

                # save the normals (3 channels)
                # shape => [3, Z, Y, X], transpose to [Z, Y, X, 3] so tifffile
                # sees it as a multi-channel 3D volume.
                normals_4d = normal_labels[batch_idx].cpu().numpy()  # shape [3, Z, Y, X]
                normals_4d = np.transpose(normals_4d, (1, 2, 3, 0))  # shape [Z, Y, X, 3]
                tifffile.imwrite("batch0_normals.tif", normals_4d)

                if use_affinities:
                    # save the affinity maps (6 channels)
                    # shape => [6, Z, Y, X]. transpose to [Z, Y, X, 6].
                    affinity_4d = affinity_labels[batch_idx].cpu().numpy()  # [6, Z, Y, X]
                    affinity_4d = np.transpose(affinity_4d, (1, 2, 3, 0))  # [Z, Y, X, 6]
                    tifffile.imwrite("batch0_affinities.tif", affinity_4d)

            inputs = inputs.to(device, dtype=torch.float32)
            sheet_labels = sheet_labels.to(device, dtype=torch.float32)
            normal_labels = normal_labels.to(device, dtype=torch.float32)
            if use_affinities:
                affinity_labels = affinity_labels.to(device, dtype=torch.float32)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                sheet_loss = sheet_criterion(outputs['sheet'], sheet_labels)
                normal_loss = normal_criterion(outputs['normals'], normal_labels)
                if use_affinities:
                    affinity_loss = affinity_criterion(outputs['affinities'], affinity_labels)
                    loss = w_sheet * sheet_loss + w_normal * normal_loss + w_affinity * affinity_loss
                else:
                    loss = w_sheet * sheet_loss + w_normal * normal_loss

            running_loss += loss.item()
            running_loss_sheet += sheet_loss.item()
            running_loss_normal += normal_loss.item()
            if use_affinities:
                running_loss_affinity += affinity_loss.item()

            scaler.scale(loss).backward()
            if (i + 1) % grad_accumulate_n == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

            steps += 1
            avg_sheet_loss = running_loss_sheet / ((i + 1) / grad_accumulate_n)
            avg_normal_loss = running_loss_normal / ((i + 1) / grad_accumulate_n)
            if use_affinities:
                avg_affinity_loss = running_loss_affinity / ((i + 1) / grad_accumulate_n)
            avg_running_loss = running_loss / ((i + 1) / grad_accumulate_n)

            pbar.set_description(f"Epoch: {epoch + 1} | Loss: {avg_running_loss:.4f} | "
                                 f"Sheet Loss: {avg_sheet_loss:.4f} | Normal Loss: {avg_normal_loss:.4f}")
        pbar.close()

        epoch_avg_sheet_loss = running_loss_sheet / len(dataloader)
        epoch_avg_normal_loss = running_loss_normal / len(dataloader)
        if use_affinities:
            epoch_avg_affinity_loss = running_loss_affinity / len(dataloader)
        epoch_avg_loss = running_loss / (len(dataloader))

        print(f"Epoch: {epoch + 1} | Sheet Loss: {epoch_avg_sheet_loss:.4f} | "
              f"Normal Loss: {epoch_avg_normal_loss:.4f}")

        print(
            f'\End of Epoch {epoch + 1} | Average Sheet Loss: {epoch_avg_sheet_loss:.4f} | Average Normal Loss: {epoch_avg_normal_loss:.4f} | Average Loss: {epoch_avg_loss}\n'
        )
        torch.save(model.state_dict(), f'{ckpt_out_base}/{model_name}_{epoch + 1}.pth')
        torch.save(optimizer.state_dict(), f'{ckpt_out_base}/{model_name}_{epoch + 1}_optimizer.pth')
        torch.save(scheduler.state_dict(), f'{ckpt_out_base}/{model_name}_{epoch + 1}_scheduler.pth')

        # ---- validation ----- #
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                val_running_loss_sheet = 0.0
                val_running_loss_normal = 0.0
                if use_affinities:
                    val_running_loss_affinity = 0.0

                pbar = tqdm(enumerate(val_dataloader), total=max_val_steps_per_epoch)
                val_steps = 0

                for i, data in pbar:
                    if i >= max_val_steps_per_epoch:
                        break

                    if use_affinities:
                        inputs, sheet_labels, normal_labels, affinity_labels = data
                    else:
                        inputs, sheet_labels, normal_labels = data

                    inputs = inputs.to(device, dtype=torch.float32)
                    sheet_labels = sheet_labels.to(device, dtype=torch.float32)
                    normal_labels = normal_labels.to(device, dtype=torch.float32)
                    if use_affinities:
                        affinity_labels = affinity_labels.to(device, dtype=torch.float32)

                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        sheet_loss = sheet_criterion(outputs['sheet'], sheet_labels)
                        normal_loss = normal_criterion(outputs['normals'], normal_labels)
                        if use_affinities:
                            affinity_loss = affinity_criterion(outputs['affinities'], affinity_labels)
                            loss = w_sheet * sheet_loss + w_normal * normal_loss + w_affinity * affinity_loss
                        else:
                            loss = w_sheet * sheet_loss + w_normal * normal_loss

                        if i == 0:
                            if use_affinities:
                                save_debug_gif(inputs, sheet_labels, normal_labels, affinity_labels, outputs, save_prefix=f"evals/img_e{epoch}")
                            else:
                                save_debug_gif(inputs, sheet_labels, normal_labels, outputs,
                                               save_prefix=f"evals/img_e{epoch}")

                    val_running_loss += loss.item()
                    val_running_loss_sheet += sheet_loss.item()
                    val_running_loss_normal += normal_loss.item()
                    if use_affinities:
                        val_running_loss_affinity += affinity_loss.item()

                    val_steps += 1

                    avg_val_running_loss = val_running_loss / (i + 1)
                    avg_val_sheet_loss = val_running_loss_sheet / (i + 1)
                    avg_val_normal_loss = val_running_loss_normal / (i + 1)
                    if use_affinities:
                        avg_val_affinity_loss = val_running_loss_affinity / (i + 1)

                    # Update progress bar
                    pbar.set_description(
                        f'Val Running Loss: {avg_val_running_loss:.8f}, Sheet Loss: {avg_val_sheet_loss:.8f}, Normal Loss: {avg_val_normal_loss:.8f}')

            pbar.close()

            epoch_val_avg_sheet_loss = val_running_loss_sheet / len(val_dataloader)
            epoch_val_avg_normal_loss = val_running_loss_normal / len(val_dataloader)
            if use_affinities:
                epoch_val_avg_affinity_loss = val_running_loss_affinity / len(val_dataloader)
            epoch_val_avg_loss = val_running_loss / len(val_dataloader)

            # Print average loss per batch after each epoch
            print(
                f'\nEnd of Epoch: {epoch + 1}, Val Average Sheet Loss: {epoch_val_avg_sheet_loss:.8f}, Val Average Normal Loss: {epoch_val_avg_normal_loss:.8f}, Val Average Loss: {epoch_val_avg_loss:.8f}\n')

            scheduler.step()

    print('Training Finished!')
    torch.save(model.state_dict(), f'{model_name}_final.pth')

if __name__ == '__main__':
    train()



# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]