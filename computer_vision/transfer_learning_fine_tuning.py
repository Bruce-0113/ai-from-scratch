import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def inspect_resnet18_backbone() -> None:
    """
    Print the pretrained ResNet18 architecture and 
    the input feature dimension of its classifier head (fc layer).
    """
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print(backbone)
    print()
    print("classifier head:", backbone.fc)
    print("feature dim:", backbone.fc.in_features)

def make_feature_extractor(num_classes=10):
    """Build a feature-extractor model from pretrained ResNet18: freeze all
    backbone parameters and replace the classifier head with a new trainable
    linear layer that outputs num_classes.

    Args:
        num_classes: Number of output classes for the new classifier head.

    Returns:
        nn.Module: Model with frozen backbone parameters and a replaced fc layer.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def inspect_feature_extractor_params():
    """
    Print the total number of trainable and frozen parameters in the feature-extractor model, 
    useful for verifying that freezing took effect.
    """
    model = make_feature_extractor(num_classes=10)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable: {trainable:>10,}")
    print(f"frozen:    {frozen:>10,}")

def discriminative_param_groups(model, base_lr=1e-3, decay=0.3) -> list:
    """Build per-stage optimizer parameter groups with discriminative learning rates.

    Earlier stages (closer to the input) get exponentially smaller learning
    rates than later stages (closer to the output), based on `decay`.

    Args:
        model: A ResNet18-like model with conv1/bn1/layer1-4/fc attributes.
        base_lr: Learning rate assigned to the last stage (fc).
        decay: Multiplicative decay applied per stage moving toward the input.

    Returns:
        list[dict]: Parameter group dicts (params, lr, name) for use with an optimizer.
    """
    stages = [
        ["conv1", "bn1"],
        ["layer1"],
        ["layer2"],
        ["layer3"],
        ["layer4"],
        ["fc"],
    ]
    groups = []
    for i, names in enumerate(stages):
        lr = base_lr * (decay ** (len(stages) - 1 - i))
        params = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) for k in names)]
        if params:
            groups.append({"params": params, "lr": lr, "name": "_".join(names)})
    return groups

def inspect_discriminative_fine_tuning_params() -> None:
    """Print the learning rate and parameter count of each discriminative
    fine-tuning stage produced by discriminative_param_groups."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for p in model.parameters():
        p.requires_grad = True

    groups = discriminative_param_groups(model)
    for g in groups:
        print(f"{g['name']:>10s}  lr={g['lr']:.2e}  params={sum(p.numel() for p in g['params']):>8,}")    

def freeze_bn_stats(model):
    """Freeze all BatchNorm layers in-place: set them to eval mode (stops
    running-stat updates) and disable gradients for their affine parameters.

    Args:
        model: The model whose BatchNorm layers should be frozen.

    Returns:
        nn.Module: The same model, with BatchNorm layers frozen.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    return model

def fine_tune(model, train_loader, val_loader, device, epochs=5, base_lr=1e-3, freeze_bn=False):
    """Fine-tune a model with discriminative learning rates and a cosine LR schedule.

    Trains with SGD (Nesterov momentum) over per-stage parameter groups from
    discriminative_param_groups, evaluates accuracy on val_loader after every
    epoch, and optionally keeps BatchNorm layers frozen throughout training.

    Args:
        model: The model to fine-tune.
        train_loader: DataLoader yielding (input, label) training batches.
        val_loader: DataLoader yielding (input, label) validation batches.
        device: Device to move the model and batches to.
        epochs: Number of training epochs.
        base_lr: Base learning rate passed to discriminative_param_groups.
        freeze_bn: If True, freeze BatchNorm layers (via freeze_bn_stats) each epoch.

    Returns:
        nn.Module: The fine-tuned model.
    """
    model = model.to(device)
    groups = discriminative_param_groups(model, base_lr=base_lr)
    optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            freeze_bn_stats(model)
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()

        model.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                va_total += x.size(0)
                va_correct += (pred == y).sum().item()
        print(f"epoch {epoch}  train {tr_loss/tr_total:.3f}/{tr_correct/tr_total:.3f}  "
              f"val {va_correct/va_total:.3f}")
    return model

