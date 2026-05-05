r"""
U-Net + DTAM (Directional Topology Attention Module)
Training script for pavement crack segmentation
Loss: BCE + Dice

Notes:
- Input size fixed at H=512, W=1024
- Supports BMP images/masks
- Masks are binary with black background / white cracks
- Uses EfficientNet-B7 encoder from segmentation_models_pytorch
- Custom U-Net style decoder implemented directly in this script
- DTAM is inserted on skip features before decoder fusion
- Uses AMP with torch.amp.autocast + torch.amp.GradScaler
- Saves:
    - best_model.pth
    - last_model.pth
    - training_log.csv
    - training_log.xlsx
"""

import os
import gc
import cv2
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# CONFIG
# =========================
SEED = 42

IMAGE_DIR  = "data/images"
MASK_DIR   = "data/masks"
OUTPUT_DIR = "outputs/DTAM_BCE_Dice"

TARGET_H = 512
TARGET_W = 1024

BATCH_SIZE = 2
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.10
NUM_WORKERS = 4

ENCODER_NAME = "efficientnet-b7"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
CLASSES = 1

PATIENCE_LR = 8
MIN_LR = 1e-7

THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()

# DTAM settings
DTAM_KERNEL_SIZE = 15
DTAM_DILATION = 2
DTAM_REDUCTION = 4

# =========================
# REPRODUCIBILITY
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# =========================
# UTILS
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_image_files(folder):
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = []
    for p in Path(folder).iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    return sorted(files)


def stem(path):
    return Path(path).stem


def find_matching_mask(image_path, mask_dir):
    image_stem = stem(image_path)
    possible_exts = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    for ext in possible_exts:
        candidate = os.path.join(mask_dir, image_stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def iou_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def precision_recall_f1_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return precision.mean(), recall.mean(), f1.mean()


def accuracy_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == targets).float().mean()
    return correct


# =========================
# DATASET
# =========================
class CrackDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        mask = (mask > 127).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float()

        return image, mask


def get_train_transforms():
    return A.Compose([
        A.Resize(height=TARGET_H, width=TARGET_W),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=8, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(height=TARGET_H, width=TARGET_W),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# =========================
# LOSS
# =========================
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        total = bce + dice_loss
        return total, bce.detach(), dice_loss.detach()


# =========================
# MODEL BLOCKS
# =========================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3, padding=1),
            ConvBNAct(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class DirectionalTopologyAttentionModule(nn.Module):
    """
    DTAM:
    - Longitudinal scanner: 15x1
    - Transverse scanner: 1x15
    - 3D texture scanner: 3x3 dilated
    - Gated fusion -> sigmoid attention map
    - Output: x * (1 + attention)
    """
    def __init__(self, channels, kernel_size=15, dilation=2, reduction=4):
        super().__init__()
        inter_channels = max(channels // reduction, 16)

        pad_v = kernel_size // 2
        pad_h = kernel_size // 2
        pad_d = dilation

        self.branch_vertical = nn.Sequential(
            ConvBNAct(channels, inter_channels, kernel_size=(kernel_size, 1), padding=(pad_v, 0)),
            ConvBNAct(inter_channels, inter_channels, kernel_size=1)
        )

        self.branch_horizontal = nn.Sequential(
            ConvBNAct(channels, inter_channels, kernel_size=(1, kernel_size), padding=(0, pad_h)),
            ConvBNAct(inter_channels, inter_channels, kernel_size=1)
        )

        self.branch_texture = nn.Sequential(
            ConvBNAct(channels, inter_channels, kernel_size=3, padding=pad_d, dilation=dilation),
            ConvBNAct(inter_channels, inter_channels, kernel_size=1)
        )

        self.fuse = nn.Sequential(
            ConvBNAct(inter_channels * 3, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.branch_vertical(x)
        h = self.branch_horizontal(x)
        t = self.branch_texture(x)

        fused = torch.cat([v, h, t], dim=1)
        attention = self.fuse(fused)

        return x * (1.0 + attention)


class DecoderBlockDTAM(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.skip_channels = skip_channels
        self.dtam = DirectionalTopologyAttentionModule(
            channels=skip_channels,
            kernel_size=DTAM_KERNEL_SIZE,
            dilation=DTAM_DILATION,
            reduction=DTAM_REDUCTION
        ) if skip_channels > 0 else None

        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            if self.dtam is not None:
                skip = self.dtam(skip)

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


class UnetDTAM(nn.Module):
    def __init__(
        self,
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        encoder_channels = self.encoder.out_channels
        # Typical encoder_channels order:
        # [input, stage1, stage2, stage3, stage4, stage5]

        self.center = DoubleConv(encoder_channels[-1], encoder_channels[-1])

        self.decoder5 = DecoderBlockDTAM(
            in_channels=encoder_channels[-1],
            skip_channels=encoder_channels[-2],
            out_channels=decoder_channels[0]
        )
        self.decoder4 = DecoderBlockDTAM(
            in_channels=decoder_channels[0],
            skip_channels=encoder_channels[-3],
            out_channels=decoder_channels[1]
        )
        self.decoder3 = DecoderBlockDTAM(
            in_channels=decoder_channels[1],
            skip_channels=encoder_channels[-4],
            out_channels=decoder_channels[2]
        )
        self.decoder2 = DecoderBlockDTAM(
            in_channels=decoder_channels[2],
            skip_channels=encoder_channels[-5],
            out_channels=decoder_channels[3]
        )
        self.decoder1 = DecoderBlockDTAM(
            in_channels=decoder_channels[3],
            skip_channels=0,
            out_channels=decoder_channels[4]
        )

        self.segmentation_head = nn.Conv2d(decoder_channels[-1], classes, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.encoder(x)

        # features order:
        # f0, f1, f2, f3, f4, f5
        f0, f1, f2, f3, f4, f5 = features

        x = self.center(f5)
        x = self.decoder5(x, f4)
        x = self.decoder4(x, f3)
        x = self.decoder3(x, f2)
        x = self.decoder2(x, f1)
        x = self.decoder1(x, None)

        logits = self.segmentation_head(x)

        if logits.shape[-2:] != (TARGET_H, TARGET_W):
            logits = F.interpolate(logits, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)

        return logits


# =========================
# DATA PREP
# =========================
def prepare_data(image_dir, mask_dir):
    image_paths = list_image_files(image_dir)

    paired_images = []
    paired_masks = []

    for image_path in image_paths:
        mask_path = find_matching_mask(image_path, mask_dir)
        if mask_path is not None:
            paired_images.append(image_path)
            paired_masks.append(mask_path)

    if len(paired_images) == 0:
        raise ValueError("No matched image-mask pairs found.")

    return paired_images, paired_masks


# =========================
# TRAIN / VALIDATE
# =========================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()

    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_acc = 0.0

    progress = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(images)
            loss, bce_loss, dice_loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            iou = iou_score_from_logits(logits, masks, threshold=THRESHOLD).item()
            precision, recall, f1 = precision_recall_f1_from_logits(logits, masks, threshold=THRESHOLD)
            acc = accuracy_from_logits(logits, masks, threshold=THRESHOLD).item()

        batch_size_now = images.size(0)
        running_loss += loss.item() * batch_size_now
        running_bce += bce_loss.item() * batch_size_now
        running_dice_loss += dice_loss.item() * batch_size_now
        running_iou += iou * batch_size_now
        running_f1 += f1.item() * batch_size_now
        running_precision += precision.item() * batch_size_now
        running_recall += recall.item() * batch_size_now
        running_acc += acc * batch_size_now

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "f1": f"{f1.item():.4f}",
            "iou": f"{iou:.4f}",
        })

    n = len(loader.dataset)
    return {
        "loss": running_loss / n,
        "bce": running_bce / n,
        "dice_loss": running_dice_loss / n,
        "iou": running_iou / n,
        "f1": running_f1 / n,
        "precision": running_precision / n,
        "recall": running_recall / n,
        "accuracy": running_acc / n,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch):
    model.eval()

    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_acc = 0.0

    progress = tqdm(loader, desc=f"Val   Epoch {epoch}", leave=False)

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(images)
            loss, bce_loss, dice_loss = criterion(logits, masks)

        iou = iou_score_from_logits(logits, masks, threshold=THRESHOLD).item()
        precision, recall, f1 = precision_recall_f1_from_logits(logits, masks, threshold=THRESHOLD)
        acc = accuracy_from_logits(logits, masks, threshold=THRESHOLD).item()

        batch_size_now = images.size(0)
        running_loss += loss.item() * batch_size_now
        running_bce += bce_loss.item() * batch_size_now
        running_dice_loss += dice_loss.item() * batch_size_now
        running_iou += iou * batch_size_now
        running_f1 += f1.item() * batch_size_now
        running_precision += precision.item() * batch_size_now
        running_recall += recall.item() * batch_size_now
        running_acc += acc * batch_size_now

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "f1": f"{f1.item():.4f}",
            "iou": f"{iou:.4f}",
        })

    n = len(loader.dataset)
    return {
        "loss": running_loss / n,
        "bce": running_bce / n,
        "dice_loss": running_dice_loss / n,
        "iou": running_iou / n,
        "f1": running_f1 / n,
        "precision": running_precision / n,
        "recall": running_recall / n,
        "accuracy": running_acc / n,
    }


# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 90)
    print("Training U-Net + DTAM with BCE + Dice")
    print("=" * 90)
    print(f"Device       : {DEVICE}")
    print(f"Image Dir    : {IMAGE_DIR}")
    print(f"Mask Dir     : {MASK_DIR}")
    print(f"Output Dir   : {OUTPUT_DIR}")
    print(f"Input Size   : H={TARGET_H}, W={TARGET_W}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 90)

    image_paths, mask_paths = prepare_data(IMAGE_DIR, MASK_DIR)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=VAL_SPLIT,
        random_state=SEED,
        shuffle=True,
    )

    print(f"Total samples : {len(image_paths)}")
    print(f"Train samples : {len(train_imgs)}")
    print(f"Val samples   : {len(val_imgs)}")

    train_dataset = CrackDataset(
        image_paths=train_imgs,
        mask_paths=train_masks,
        transforms=get_train_transforms()
    )

    val_dataset = CrackDataset(
        image_paths=val_imgs,
        mask_paths=val_masks,
        transforms=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = UnetDTAM(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        decoder_channels=(256, 128, 64, 32, 16),
    ).to(DEVICE)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=PATIENCE_LR,
        min_lr=MIN_LR,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_val_f1 = -1.0
    history = []

    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    last_model_path = os.path.join(OUTPUT_DIR, "last_model.pth")
    csv_log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    xlsx_log_path = os.path.join(OUTPUT_DIR, "training_log.xlsx")

    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=DEVICE,
            epoch=epoch
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            epoch=epoch
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["f1"])

        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,

            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_iou": train_metrics["iou"],
            "train_f1": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_accuracy": train_metrics["accuracy"],

            "val_loss": val_metrics["loss"],
            "val_bce": val_metrics["bce"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        df = pd.DataFrame(history)
        df.to_csv(csv_log_path, index=False)
        df.to_excel(xlsx_log_path, index=False)

        improved = val_metrics["f1"] > best_val_f1
        if improved:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "config": {
                    "encoder_name": ENCODER_NAME,
                    "encoder_weights": ENCODER_WEIGHTS,
                    "target_h": TARGET_H,
                    "target_w": TARGET_W,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "num_epochs": NUM_EPOCHS,
                    "loss": "BCE + Dice",
                    "module": "DTAM",
                }
            }, best_model_path)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "config": {
                "encoder_name": ENCODER_NAME,
                "encoder_weights": ENCODER_WEIGHTS,
                "target_h": TARGET_H,
                "target_w": TARGET_W,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "loss": "BCE + Dice",
                "module": "DTAM",
            }
        }, last_model_path)

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS:03d}] | "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"{'BEST' if improved else ''}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("U-Net + DTAM Training Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Val F1: {best_val_f1:.6f}\n")
        f.write(f"Total Training Time (sec): {total_time:.2f}\n")
        f.write(f"Total Training Time (hr): {total_time / 3600:.2f}\n")
        f.write(f"Best Model Path: {best_model_path}\n")
        f.write(f"Last Model Path: {last_model_path}\n")
        f.write(f"CSV Log Path: {csv_log_path}\n")
        f.write(f"XLSX Log Path: {xlsx_log_path}\n")

    print("=" * 90)
    print("Training completed.")
    print(f"Best Val F1: {best_val_f1:.6f}")
    print(f"Best model : {best_model_path}")
    print(f"Last model : {last_model_path}")
    print(f"CSV log    : {csv_log_path}")
    print(f"XLSX log   : {xlsx_log_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()