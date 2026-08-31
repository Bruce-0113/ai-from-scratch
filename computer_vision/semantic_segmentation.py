"""U-Net for semantic segmentation, trained from scratch.

Covers a small U-Net encoder-decoder architecture, a combined
cross-entropy + soft-Dice loss, per-class IoU evaluation, and a synthetic
shapes dataset used to train and sanity-check the whole pipeline end to end.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two (3x3 conv -> BatchNorm -> ReLU) blocks, the basic U-Net building block."""

    def __init__(self, in_c, out_c):
        """
        Args:
            in_c: number of input channels.
            out_c: number of output channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (N, in_c, H, W) input feature map.

        Returns:
            (N, out_c, H, W) feature map.
        """
        return self.net(x)


class Down(nn.Module):
    """Downscaling step: 2x max-pool followed by a DoubleConv."""

    def __init__(self, in_c, out_c):
        """
        Args:
            in_c: number of input channels.
            out_c: number of output channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

    def forward(self, x):
        """
        Args:
            x: (N, in_c, H, W) input feature map.

        Returns:
            (N, out_c, H/2, W/2) feature map.
        """
        return self.net(x)


class Up(nn.Module):
    """Upscaling step: bilinear upsample, concat with the encoder skip, then DoubleConv."""

    def __init__(self, in_c, out_c):
        """
        Args:
            in_c: channels after concatenating the upsampled input with the
                skip connection (upsampled channels + skip channels).
            out_c: number of output channels.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip):
        """
        Args:
            x: (N, C, H, W) feature map from the previous, lower-resolution
                decoder stage.
            skip: (N, C_skip, H', W') encoder feature map to concatenate, at
                the target resolution.

        Returns:
            (N, out_c, H', W') feature map.
        """
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Small U-Net for semantic segmentation with 4 down/up stages."""

    def __init__(self, in_channels=3, num_classes=2, base=64):
        """
        Args:
            in_channels: number of input image channels.
            num_classes: number of segmentation classes to predict.
            base: number of channels in the first encoder stage; doubles at
                each downsampling step.
        """
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16 + base * 8, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u3 = Up(base * 4 + base * 2, base * 2)
        self.u4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (N, in_channels, H, W) input image batch.

        Returns:
            (N, num_classes, H, W) per-pixel class logits.
        """
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Losses & metrics
# ---------------------------------------------------------------------------

def dice_loss(logits, targets, num_classes, eps=1e-6):
    """Soft Dice loss averaged over classes.

    Args:
        logits: (N, num_classes, H, W) raw network output.
        targets: (N, H, W) int64 ground-truth class indices.
        num_classes: number of segmentation classes.
        eps: small constant to avoid division by zero.

    Returns:
        Scalar tensor: 1 - mean per-class Dice coefficient.
    """
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, num_classes, lam=1.0):
    """Cross-entropy plus a weighted Dice term.

    Args:
        logits: (N, num_classes, H, W) raw network output.
        targets: (N, H, W) int64 ground-truth class indices.
        num_classes: number of segmentation classes.
        lam: weight applied to the Dice loss term.

    Returns:
        total: scalar tensor, ce + lam * dice.
        parts: dict of the unweighted per-term loss values (floats), for logging.
    """
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce.item(), "dice": dc.item()}


@torch.no_grad()
def iou_per_class(logits, targets, num_classes):
    """Intersection-over-union for each class.

    Args:
        logits: (N, num_classes, H, W) raw network output.
        targets: (N, H, W) int64 ground-truth class indices.
        num_classes: number of segmentation classes.

    Returns:
        (num_classes,) float tensor of per-class IoU; a class absent from
        both the prediction and target is reported as NaN.
    """
    preds = logits.argmax(dim=1)
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious[c] = (inter / union) if union > 0 else torch.tensor(float("nan"))
    return ious


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def synthetic_segmentation(num_samples=200, size=64, seed=0):
    """Generate a toy segmentation dataset of circles/squares on flat backgrounds.

    Each image gets a random background color with 1-3 random shapes
    (class 1: circle, class 2: square) painted on it, plus a little Gaussian
    noise.

    Args:
        num_samples: number of images to generate.
        size: height/width of each (square) image, in pixels.
        seed: random seed for reproducibility.

    Returns:
        images: (num_samples, size, size, 3) float32 ndarray in [0, 1].
        masks: (num_samples, size, size) int64 ndarray of class indices
            (0=background, 1=circle, 2=square).
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    for i in range(num_samples):
        bg = rng.uniform(0, 1, (3,))
        images[i] = bg
        masks[i] = 0
        num_shapes = rng.integers(1, 4)
        for _ in range(num_shapes):
            cls = int(rng.integers(1, 3))
            color = rng.uniform(0, 1, (3,))
            cx, cy = rng.integers(10, size - 10, size=2)
            r = int(rng.integers(4, 12))
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            if cls == 1:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            else:
                mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            images[i][mask] = color
            masks[i][mask] = cls
        images[i] += rng.normal(0, 0.02, images[i].shape)
        images[i] = np.clip(images[i], 0, 1)
    return images, masks


class SegDataset(Dataset):
    """Wraps `synthetic_segmentation`-style arrays as a torch Dataset."""

    def __init__(self, images, masks):
        """
        Args:
            images: (N, H, W, 3) float32 ndarray of images in [0, 1].
            masks: (N, H, W) int64 ndarray of per-pixel class indices.
        """
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        """
        Returns:
            img: (3, H, W) float32 tensor.
            mask: (H, W) int64 tensor.
        """
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[i]).long()
        return img, mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, num_classes):
    """Run one training epoch.

    Args:
        model: segmentation model mapping (N, C, H, W) images to
            (N, num_classes, H, W) logits.
        loader: DataLoader yielding (images, masks) batches.
        optimizer: optimizer over `model`'s parameters.
        device: torch device to run on.
        num_classes: number of segmentation classes.

    Returns:
        avg_loss: mean combined loss over the epoch, weighted by batch size.
        avg_iou: (num_classes,) tensor of mean per-class IoU over the
            epoch's batches.
    """
    model.train()
    loss_sum, total = 0.0, 0
    iou_sum = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, _ = combined_loss(logits, y, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        iou_sum += iou_per_class(logits, y, num_classes).nan_to_num(0)
    return loss_sum / total, iou_sum / len(loader)


def run_demo():
    """Build a small U-Net, smoke-test a forward pass, then train on synthetic data."""
    torch.manual_seed(0)
    num_classes = 3  # background, circle, square (see synthetic_segmentation)

    model = UNet(in_channels=3, num_classes=num_classes, base=32)
    x = torch.randn(1, 3, 256, 256)
    print(f"output: {model(x).shape}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    images, masks = synthetic_segmentation(num_samples=200, size=64, seed=0)
    loader = DataLoader(SegDataset(images, masks), batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        loss, iou = train_one_epoch(model, loader, optimizer, device, num_classes)
        iou_str = ", ".join(f"{v:.3f}" for v in iou.tolist())
        print(f"epoch {epoch + 1}: loss={loss:.4f} iou=[{iou_str}]")


if __name__ == "__main__":
    run_demo()
