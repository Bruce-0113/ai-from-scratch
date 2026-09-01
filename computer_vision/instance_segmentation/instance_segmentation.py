"""
Instance segmentation experiments with Mask R-CNN.

This script walks through the core building blocks of Mask R-CNN:
1. A from-scratch RoIAlign implementation, checked against torchvision's
   official `roi_align` for correctness.
2. Loading a pretrained Mask R-CNN model and running inference to inspect
   its outputs (boxes, labels, scores, masks).
3. Customizing the model heads for a different number of classes.
4. Freezing the backbone/FPN for fine-tuning with fewer trainable params.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def roi_align_single(feature, box, output_size=7, spatial_scale=1 / 16.0):
    """
    Crop and resize a region of a feature map into a fixed-size grid using
    bilinear-sampled RoIAlign (no quantization of the RoI boundaries).

    Args:
        feature: (C, H, W) single-image feature map.
        box: (x1, y1, x2, y2) region of interest in original image pixel
            coordinates.
        output_size: side length of the output grid (7 for box head, 14 for
            mask head).
        spatial_scale: reciprocal of the feature map stride, used to map the
            box from image coordinates into feature map coordinates.

    Returns:
        (C, output_size, output_size) tensor of pooled features.
    """
    C, H, W = feature.shape
    x1, y1, x2, y2 = [c * spatial_scale - 0.5 for c in box]
    bin_w = (x2 - x1) / output_size
    bin_h = (y2 - y1) / output_size

    grid_y = torch.linspace(y1 + bin_h / 2, y2 - bin_h / 2, output_size)
    grid_x = torch.linspace(x1 + bin_w / 2, x2 - bin_w / 2, output_size)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    gx = 2 * (xx + 0.5) / W - 1
    gy = 2 * (yy + 0.5) / H - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                            align_corners=False)
    return sampled.squeeze(0)


feature = torch.randn(1, 16, 50, 50)
boxes = torch.tensor([[0, 10, 20, 100, 90]], dtype=torch.float32)  # (batch_idx, x1, y1, x2, y2)

ours = roi_align_single(feature[0], boxes[0, 1:].tolist(), output_size=7, spatial_scale=1/4)
theirs = roi_align(feature, boxes, output_size=(7, 7), spatial_scale=1/4, sampling_ratio=1, aligned=True)[0]

print(f"shape ours:   {tuple(ours.shape)}")
print(f"shape theirs: {tuple(theirs.shape)}")
print(f"max|diff|:    {(ours - theirs).abs().max().item():.3e}")


model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
print(f"classes (including background): {len(model.roi_heads.box_predictor.cls_score.out_features * [0])}")


with torch.no_grad():
    x = torch.randn(3, 400, 600)
    predictions = model([x])
p = predictions[0]
print(f"boxes:  {tuple(p['boxes'].shape)}")
print(f"labels: {tuple(p['labels'].shape)}")
print(f"scores: {tuple(p['scores'].shape)}")
print(f"masks:  {tuple(p['masks'].shape)}")

binary_masks = (p['masks'] > 0.5).squeeze(1)  # (N, H, W) boolean


def build_custom_maskrcnn(num_classes):
    """
    Build a pretrained Mask R-CNN and swap its box and mask prediction heads
    to match a custom number of classes (transfer learning setup).

    Args:
        num_classes: number of output classes, including background.

    Returns:
        A Mask R-CNN model with reinitialized box and mask predictors.
    """
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

custom = build_custom_maskrcnn(num_classes=5)
print(f"custom cls_score.out_features: {custom.roi_heads.box_predictor.cls_score.out_features}")


def freeze_backbone_and_fpn(model):
    """
    Freeze the backbone (ResNet) and FPN parameters in place so only the
    RPN and RoI heads remain trainable, reducing the fine-tuning cost.

    Args:
        model: a Mask R-CNN model.

    Returns:
        The same model, with `requires_grad=False` on backbone/FPN params.
    """
    # torchvision Mask R-CNN packs the FPN inside `model.backbone` (as
    # `model.backbone.fpn`), so iterating `model.backbone.parameters()` covers
    # both the ResNet feature layers and the FPN lateral/output convs.
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model

custom = freeze_backbone_and_fpn(custom)
trainable = sum(p.numel() for p in custom.parameters() if p.requires_grad)
print(f"trainable after freeze: {trainable:,}")