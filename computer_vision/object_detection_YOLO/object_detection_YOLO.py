"""Minimal from-scratch YOLO-style single-stage object detector.

Covers the pieces needed to train and run one detection head: box IoU/NMS
utilities, anchor-based coordinate encode/decode, the conv detection head,
per-image target assignment, the multi-part training loss, and inference
postprocessing (decode + NMS).
"""

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Box utilities
# --------------------------------------------------------------------------

def box_iou(boxes_a, boxes_b):
    """Pairwise IoU between two sets of axis-aligned boxes.

    Args:
        boxes_a: (N, 4) ndarray of boxes in (x1, y1, x2, y2) format.
        boxes_b: (M, 4) ndarray of boxes in (x1, y1, x2, y2) format.

    Returns:
        (N, M) ndarray where entry [i, j] is the IoU of boxes_a[i] and boxes_b[j].
    """
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)


def nms(boxes, scores, iou_threshold=0.45):
    """Greedy non-maximum suppression.

    Args:
        boxes: (N, 4) ndarray of boxes in (x1, y1, x2, y2) format.
        scores: (N,) ndarray of confidence scores, one per box.
        iou_threshold: boxes overlapping a kept box above this IoU are suppressed.

    Returns:
        int64 ndarray of indices into `boxes`/`scores` to keep, highest score first.
    """
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)


# --------------------------------------------------------------------------
# Anchor-relative coordinate encode/decode
# --------------------------------------------------------------------------

def sigmoid(x):
    """Elementwise logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    """Encode a ground-truth box into YOLO's per-cell/anchor offset targets.

    Args:
        box_xyxy: (x1, y1, x2, y2) box in image pixel coordinates.
        cell_x: grid column index of the responsible cell.
        cell_y: grid row index of the responsible cell.
        stride: pixels per grid cell (image_size / grid_size).
        anchor_wh: (w, h) of the responsible anchor box, in pixels.

    Returns:
        ndarray [tx, ty, tw, th] of raw regression targets.
    """
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    """Inverse of `encode`: convert raw (tx, ty, tw, th) offsets back to a box.

    Args:
        tx_ty_tw_th: (tx, ty, tw, th) raw regression values (tx/ty pre-sigmoid).
        cell_x: grid column index of the cell the prediction came from.
        cell_y: grid row index of the cell the prediction came from.
        stride: pixels per grid cell (image_size / grid_size).
        anchor_wh: (w, h) of the anchor box, in pixels.

    Returns:
        ndarray [x1, y1, x2, y2] box in image pixel coordinates.
    """
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class YOLOHead(nn.Module):
    """Detection head: a 1x1 conv predicting per-anchor box + objectness + class scores."""

    def __init__(self, in_c, num_anchors, num_classes):
        """
        Args:
            in_c: number of input feature-map channels.
            num_anchors: number of anchor boxes per grid cell.
            num_classes: number of object classes.
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (N, in_c, H, W) feature map.

        Returns:
            (N, H, W, num_anchors, 5 + num_classes) tensor of raw predictions,
            where the last dim is [tx, ty, tw, th, objectness, class_logits...].
        """
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y


# --------------------------------------------------------------------------
# Training target assignment
# --------------------------------------------------------------------------

def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    """Build the dense per-cell/anchor training targets for one image.

    Each ground-truth box is assigned to the grid cell containing its center
    and to whichever anchor best matches its width/height (by IoU computed
    on width/height alone, ignoring position).

    Args:
        boxes_xyxy: iterable of (x1, y1, x2, y2) ground-truth boxes, in pixels.
        classes: iterable of integer class indices, one per box.
        anchors: iterable of (w, h) anchor box sizes, in pixels.
        stride: pixels per grid cell (image_size / grid_size).
        grid_size: number of cells per side of the (square) grid.
        num_classes: number of object classes.

    Returns:
        target: (grid_size, grid_size, num_anchors, 5 + num_classes) float32
            ndarray of [tx, ty, tw, th, 1.0, one_hot_class...] at assigned
            cells, zero elsewhere.
        has_obj: (grid_size, grid_size, num_anchors) bool ndarray marking
            which cell/anchor slots received an assignment.
    """
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    """YOLOv1/v2-style multi-part detection loss.

    Combines box regression (object cells only), objectness BCE (split into
    positive/negative cells so each can be weighted independently), and
    classification BCE (object cells only).

    Args:
        pred: (grid, grid, num_anchors, 5 + num_classes) raw network output,
            same layout as `YOLOHead.forward`'s output for one image.
        target: target array from `assign_targets`.
        has_obj: has_obj array from `assign_targets`.
        lambda_coord: weight on the box-regression term.
        lambda_obj: weight on the objectness term for object cells.
        lambda_noobj: weight on the objectness term for empty cells.
        lambda_cls: weight on the classification term.

    Returns:
        total: scalar tensor, the weighted sum of all loss terms.
        parts: dict of the unweighted per-term loss values (floats), for logging.
    """
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}


# --------------------------------------------------------------------------
# Inference postprocessing
# --------------------------------------------------------------------------

def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    """Decode raw head output for a single image into final detections.

    Filters predictions by confidence (objectness * best class prob), decodes
    surviving boxes to pixel coordinates, then applies NMS.

    Args:
        pred_tensor: (1, grid, grid, num_anchors, 5 + num_classes) raw output
            for a single image, as produced by `YOLOHead.forward`.
        anchors: iterable of (w, h) anchor box sizes, in pixels.
        stride: pixels per grid cell (image_size / grid_size).
        img_size: (H, W) of the source image; currently unused, reserved for
            optional box clipping.
        conf_threshold: minimum confidence to keep a candidate detection.
        iou_threshold: IoU threshold passed to `nms`.

    Returns:
        boxes: (K, 4) ndarray of kept boxes in (x1, y1, x2, y2) format.
        scores: (K,) ndarray of confidence scores.
        classes: (K,) int ndarray of predicted class indices.
    """
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
