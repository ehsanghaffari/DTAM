"""
Inference script for U-Net + DTAM (BCE + Dice)

Loads trained best_model.pth and runs inference on all images in INPUT_DIR.
Saves:
- predicted binary masks
- red crack overlays
- probability maps

Important:
- Network input size is fixed at H=512, W=1024
- Output masks are resized back to original image size before saving
- Designed for BMP images but also supports png/jpg/tif
"""

import os
import gc
import csv
import cv2
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================
INPUT_DIR =  "data/images"
MODEL_PATH = "model.pth"
OUTPUT_DIR = "outputs"

TARGET_H = 512
TARGET_W = 1024

ENCODER_NAME = "efficientnet-b7"
ENCODER_WEIGHTS = None
IN_CHANNELS = 3
CLASSES = 1

THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DTAM settings
DTAM_KERNEL_SIZE = 15
DTAM_DILATION = 2
DTAM_REDUCTION = 4

# Output options
SAVE_MASKS = True
SAVE_OVERLAYS = True
SAVE_PROB_MAPS = True

MASK_EXT = ".png"
OVERLAY_EXT = ".png"
PROB_EXT = ".png"

# Overlay settings
OVERLAY_ALPHA = 0.50
DRAW_ONLY_CRACK_PIXELS = True


# =========================================================
# UTILS
# =========================================================
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


# =========================================================
# MODEL BLOCKS
# =========================================================
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
        encoder_weights=None,
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


# =========================================================
# PREPROCESS
# =========================================================
transform = A.Compose([
    A.Resize(height=TARGET_H, width=TARGET_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


# =========================================================
# INFERENCE HELPERS
# =========================================================
def load_model(model_path, device):
    model = UnetDTAM(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        decoder_channels=(256, 128, 64, 32, 16),
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def preprocess_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    image = transformed["image"]

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor


def make_overlay(original_bgr, binary_mask):
    overlay = original_bgr.copy()

    red_layer = np.zeros_like(original_bgr, dtype=np.uint8)
    red_layer[:, :, 2] = 255  # BGR → red channel

    if DRAW_ONLY_CRACK_PIXELS:
        mask_bool = binary_mask > 0
        src1 = np.ascontiguousarray(original_bgr[mask_bool])
        src2 = np.ascontiguousarray(red_layer[mask_bool])
        blended = cv2.addWeighted(src1, 1.0 - OVERLAY_ALPHA, src2, OVERLAY_ALPHA, 0)
        if blended is not None:
            overlay[mask_bool] = blended
    else:
        full = cv2.addWeighted(original_bgr, 1.0 - OVERLAY_ALPHA, red_layer, OVERLAY_ALPHA, 0)
        if full is not None:
            overlay[binary_mask > 0] = full[binary_mask > 0]

    return overlay


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    overlay_dir = os.path.join(OUTPUT_DIR, "overlays")
    prob_dir = os.path.join(OUTPUT_DIR, "probability_maps")

    if SAVE_MASKS:
        ensure_dir(mask_dir)
    if SAVE_OVERLAYS:
        ensure_dir(overlay_dir)
    if SAVE_PROB_MAPS:
        ensure_dir(prob_dir)

    print("=" * 90)
    print("U-Net + DTAM Inference")
    print("=" * 90)
    print(f"Device     : {DEVICE}")
    print(f"Input Dir  : {INPUT_DIR}")
    print(f"Model Path : {MODEL_PATH}")
    print(f"Output Dir : {OUTPUT_DIR}")
    print(f"Threshold  : {THRESHOLD}")
    print("=" * 90)

    image_paths = list_image_files(INPUT_DIR)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {INPUT_DIR}")

    model = load_model(MODEL_PATH, DEVICE)

    # GPU warm-up: eliminates CUDA initialization overhead from the first image's timing
    if DEVICE == "cuda":
        dummy = torch.zeros(1, IN_CHANNELS, TARGET_H, TARGET_W, device=DEVICE)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        del dummy

    csv_rows = []
    total_wall_start = time.time()

    for image_path in tqdm(image_paths, desc="Running inference"):
        image_name = Path(image_path).name
        image_stem = Path(image_path).stem

        original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        h0, w0 = original_bgr.shape[:2]

        # ----- start per-image timer (preprocess + forward + postprocess) -----
        input_tensor = preprocess_image(original_bgr).to(DEVICE, non_blocking=True)

        # Synchronize before starting timer so any async GPU ops from prior
        # iteration don't bleed into this image's measurement
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)

        # Synchronize after forward pass so GPU finishes before stopping timer
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        prob_map = probs.squeeze().detach().cpu().numpy()
        binary_mask = (prob_map > THRESHOLD).astype(np.uint8) * 255

        # Resize back to original image size (post-processing, included in time)
        prob_map_resized = cv2.resize(prob_map, (w0, h0), interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

        t1 = time.time()
        # ----- end per-image timer (disk I/O excluded) -----

        elapsed = t1 - t0

        if SAVE_MASKS:
            mask_path = os.path.join(mask_dir, image_stem + MASK_EXT)
            cv2.imwrite(mask_path, binary_mask_resized)

        if SAVE_PROB_MAPS:
            prob_uint8 = np.clip(prob_map_resized * 255.0, 0, 255).astype(np.uint8)
            prob_path = os.path.join(prob_dir, image_stem + PROB_EXT)
            cv2.imwrite(prob_path, prob_uint8)

        if SAVE_OVERLAYS:
            overlay = make_overlay(original_bgr, binary_mask_resized)
            overlay_path = os.path.join(overlay_dir, image_stem + OVERLAY_EXT)
            cv2.imwrite(overlay_path, overlay)

        csv_rows.append({
            "image": image_name,
            "orig_h": h0,
            "orig_w": w0,
            "net_h": TARGET_H,
            "net_w": TARGET_W,
            "time_sec": round(elapsed, 6),
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_wall_time = time.time() - total_wall_start
    times = [r["time_sec"] for r in csv_rows]
    avg_time = sum(times) / max(len(times), 1)

    # Save per-image log as CSV (cleaner and machine-readable than txt)
    csv_log_path = os.path.join(OUTPUT_DIR, "inference_log.csv")
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "orig_h", "orig_w", "net_h", "net_w", "time_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        # Append summary row
        writer.writerow({
            "image": "__SUMMARY__",
            "orig_h": "",
            "orig_w": "",
            "net_h": TARGET_H,
            "net_w": TARGET_W,
            "time_sec": f"total={round(total_wall_time,4)}s | avg={round(avg_time,6)}s | n={len(times)}",
        })

    print("=" * 90)
    print("Inference completed.")
    print(f"Processed images : {len(times)}")
    print(f"Total wall time  : {total_wall_time:.4f} sec")
    print(f"Average/image    : {avg_time:.4f} sec  (preprocess + forward + postprocess)")
    print(f"Results saved to : {OUTPUT_DIR}")
    print(f"Log saved to     : {csv_log_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
