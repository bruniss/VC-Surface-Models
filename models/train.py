from pytorch3dunet.unet3d.model import MultiTaskResidualUNetSE3D
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW, SGD
import torch.nn as nn
import numpy as np
from models.dataloading.dataset import ZarrSegmentationDataset3D
from pathlib import Path
from models.visualization.plotting import save_debug_gif,log_predictions_as_video
import tifffile
from torch.utils.data import DataLoader
from models.losses.losses import masked_cosine_loss, BCEWithLogitsLossLabelSmoothing
from pytorch3dunet.unet3d.losses import BCEDiceLoss
import json
import os
from typing import List, Dict, Union, Tuple
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from models.transforms.normalization import ct_normalize





class BaseTrainer:
    """
            model_name: what you want the model to be called, and what the epochs are saved as
            patch_size: size of the patches to be cropped from the volume
            f_maps: number of feature maps in each level of the UNet
            n_levels: number of levels in the UNet
            ignore_label: value in the label volume that we will not compute losses against
            loss_only_on_label: if True, only compute loss against labeled regions
            label_smoothing: soften the loss on the label by a percentage
            min_labeled_ratio: minimum ratio of labeled pixels in a patch to be considered valid
            min_bbox_percent: minimum area a bounding box containing _all_ the labels must occupy, as a percentage of the patch size
            use_cache: if True, use a cache file to save the valid patches so you dont have to recompute
            cache_file: path to the cache file, if it exists, and also the path the new one will save to
            tasks: dictionary of tasks to be used in the model, each task must contain the number of channels and the activation type
            """

    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            config = json.load(f)

        tr_params = SimpleNamespace(**config["tr_params"])
        model_config = SimpleNamespace(**config["model_config"])
        dataset_config = SimpleNamespace(**config["dataset_config"])

        # --- configs --- #
        self.model_name = getattr(tr_params, "model_name", "SheetNorm")
        self.patch_size = tuple(getattr(tr_params, "patch_size", [192, 192, 192]))
        self.batch_size = int(getattr(tr_params, "batch_size", 2))
        self.gradient_accumulation = int(getattr(tr_params, "gradient_accumulation", 1))
        self.optimizer = str(getattr(tr_params, "optimizer", "AdamW"))
        self.tr_val_split = float(getattr(tr_params, "tr_val_split", 0.95))
        self.f_maps = list(getattr(model_config, "f_maps", [32, 64, 128, 256]))
        self.num_levels = int(getattr(model_config, "n_levels", 6))
        self.ignore_label = getattr(tr_params, "ignore_label", None)
        self.loss_only_on_label = bool(getattr(tr_params, "loss_only_on_label", False))
        self.label_smoothing = float(getattr(tr_params, "label_smoothing", 0.2))
        self.min_labeled_ratio = float(getattr(tr_params, "min_labeled_ratio", 0.1))
        self.min_bbox_percent = float(getattr(tr_params, "min_bbox_percent", 0.95))
        self.use_cache = bool(getattr(dataset_config, "use_cache", True))
        self.cache_file = Path((getattr(dataset_config, "cache_file",'valid_patches.json')))
        self.max_steps_per_epoch = int(getattr(tr_params, "max_steps_per_epoch", 500))
        self.max_val_steps_per_epoch = int(getattr(tr_params, "max_val_steps_per_epoch", 25))
        self.max_epoch = int(getattr(tr_params, "max_epoch", 500))
        self.initial_lr = float(getattr(tr_params, "initial_lr", 1e-3))
        self.weight_decay = float(getattr(tr_params, "weight_decay", 1e-4))
        self.ckpt_out_base = Path(getattr(tr_params, "ckpt_out_base", "./checkpoints/"))
        self.checkpoint_path = getattr(tr_params, "checkpoint_path", None)
        self.num_dataloader_workers = int(getattr(tr_params, "num_dataloader_workers", 4))
        self.tensorboard_log_dir = str(getattr(tr_params, "tensorboard_log_dir", "./tensorboard_logs/"))
        self.loss_fns = {} # these get defined later

        self.normalization = str(getattr(dataset_config, "normalization", "ct_normalize"))

        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        else:
            self.checkpoint_path = None

        os.makedirs(self.ckpt_out_base, exist_ok=True)
        self.data_path = Path(getattr(dataset_config, "data_path", None))
        targets = dataset_config.targets
        self.tasks = {
            key: {
                "dataset_path": val["dataset_path"],
                "channels": val["channels"],
                "activation": val["activation"]
            }
            for key, val in targets.items()
        }


    def train(self):

        model = MultiTaskResidualUNetSE3D(
            in_channels=1,
            tasks=self.tasks,
            f_maps=self.f_maps,
            num_levels=self.num_levels
        )

        dataset = ZarrSegmentationDataset3D(
            data_path=Path(self.data_path),
            patch_size=self.patch_size,
            targets=self.tasks,
            min_labeled_ratio=self.min_labeled_ratio,
            min_bbox_percent = self.min_bbox_percent,
            normalization = self.normalization,
            dilate_label = False,
            transforms = None,
            use_cache = self.use_cache,
            cache_file=Path(self.cache_file)
        )

        device = torch.device('cuda')
        model = torch.compile(model)
        model = model.to(device)

        # --- losses ---- #

        self.loss_fns = {}
        for task_name, task_info in self.tasks.items():

            # i want to use cosine loss specifically for normals
            if task_name in ("normal", "normals"):
                self.loss_fns[task_name] = masked_cosine_loss

            else:
                self.loss_fns[task_name] = BCEWithLogitsLossLabelSmoothing(smoothing=self.label_smoothing)

        if self.optimizer == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=self.initial_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay
            )

        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=self.max_epoch,
                                      eta_min=0)


        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_val_split = self.tr_val_split
        split = int(np.floor(train_val_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        batch_size = self.batch_size

        # apply gradient accumulation
        grad_accumulate_n = self.gradient_accumulation

        scaler = torch.cuda.amp.GradScaler()

        # ---- dataloading ----- #

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_indices),
                                pin_memory=True,
                                num_workers=self.num_dataloader_workers)
        val_dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=True,
                                    num_workers=self.num_dataloader_workers)

        start_epoch = 0
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            start_epoch = checkpoint['epoch'] + 1

            print(f"Resuming training from epoch {start_epoch + 1}")

        writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        global_step = 0
        # ---- training! ----- #
        for epoch in range(start_epoch, self.max_epoch):
            model.train()

            train_running_losses = {t_name: 0.0 for t_name in self.tasks}
            pbar = tqdm(enumerate(dataloader), total=self.max_steps_per_epoch)
            steps = 0

            for i, data_dict in pbar:
                if i >= self.max_steps_per_epoch:
                    break

                global_step += 1

                inputs = data_dict["image"].to(device, dtype=torch.float32)
                targets_dict = {
                    k: v.to(device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k != "image"
                }



                # forward
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    total_loss = 0.0
                    per_task_losses = {}

                    for t_name, t_gt in targets_dict.items():
                        t_pred = outputs[t_name]
                        t_loss_fn = self.loss_fns[t_name]
                        t_loss = t_loss_fn(t_pred, t_gt)

                        total_loss += t_loss
                        # Accumulate
                        train_running_losses[t_name] += t_loss.item()

                        # Also store the *current batch* loss for that task
                        per_task_losses[t_name] = t_loss.item()

                # backward
                scaler.scale(total_loss).backward()  # <-- FIXED (use total_loss instead of loss)
                if (i + 1) % grad_accumulate_n == 0 or (i + 1) == len(dataloader):

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()

                steps += 1

                writer.add_scalar("train/total_loss", total_loss.item(), global_step)
                for t_name, t_loss_value in per_task_losses.items():
                    writer.add_scalar(f"train/{t_name}_loss", t_loss_value, global_step)

                desc_parts = []
                for t_name in self.tasks:
                    avg_t_loss = train_running_losses[t_name] / steps
                    desc_parts.append(f"{t_name}: {avg_t_loss:.4f}")

                desc_str = f"Epoch {epoch + 1} => " + " | ".join(desc_parts)
                pbar.set_description(desc_str)

            pbar.close()

            for t_name in self.tasks:
                epoch_avg = train_running_losses[t_name] / steps
                print(f"Task '{t_name}', epoch {epoch + 1} avg train loss: {epoch_avg:.4f}")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch  # optionally store the current epoch so you can resume
            }, f"{self.ckpt_out_base}/{self.model_name}_{epoch + 1}.pth")

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_running_losses = {t_name: 0.0 for t_name in self.tasks}
                    val_steps = 0

                    pbar = tqdm(enumerate(val_dataloader), total=self.max_val_steps_per_epoch)
                    for i, data_dict in pbar:
                        if i >= self.max_val_steps_per_epoch:
                            break

                        inputs = data_dict["image"].to(device, dtype=torch.float32)
                        targets_dict = {
                            k: v.to(device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k != "image"
                        }

                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            total_val_loss = 0.0
                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                t_loss_fn = self.loss_fns[t_name]
                                t_loss = t_loss_fn(t_pred, t_gt)

                                total_val_loss += t_loss
                                val_running_losses[t_name] += t_loss.item()

                            val_steps += 1

                            if i == 0:
                                # We'll log a video for each task's prediction + label
                                log_predictions_as_video(
                                    writer,
                                    inputs,
                                    targets_dict,
                                    outputs,
                                    epoch
                                )

                    desc_parts = []
                    for t_name in self.tasks:
                        avg_loss_for_t = val_running_losses[t_name] / val_steps
                        desc_parts.append(f"{t_name} {avg_loss_for_t:.4f}")
                    desc_str = "Val: " + " | ".join(desc_parts)
                    pbar.set_description(desc_str)

                pbar.close()

                # Final avg for each task
                for t_name in self.tasks:
                    val_avg = val_running_losses[t_name] / val_steps
                    print(f"Task '{t_name}', epoch {epoch + 1} avg val loss: {val_avg:.4f}")

            scheduler.step()

        print('Training Finished!')
        torch.save(model.state_dict(), f'{self.model_name}_final.pth')

if __name__ == '__main__':
    trainer = BaseTrainer("/home/sean/Documents/GitHub/VC-Surface-Models/models/tasks/sheet_normals.json")
    trainer.train()



# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]