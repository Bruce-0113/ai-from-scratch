"""LoRA (Low-Rank Adaptation) and a simplified NF4-style quantization demo.

Implements LoRA injection/merging for nn.Linear layers, a toy block-wise
quantization scheme inspired by QLoRA's NF4, and a minimal training loop
to demonstrate parameter-efficient fine-tuning end to end.
"""

import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-rank update `(x @ A @ B) * scaling` added to a frozen linear layer."""

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling


class LinearWithLoRA(nn.Module):
    """Wraps a frozen `nn.Linear` and adds a trainable LoRA branch."""

    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def inject_lora(model, target_modules, rank=8, alpha=16):
    """Freeze all model parameters and replace matching `nn.Linear` layers with `LinearWithLoRA`.

    Args:
        model: The model to modify in place.
        target_modules: Substrings matched against module names to select which
            linear layers get a LoRA adapter.
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Dict mapping original module name to the new `LinearWithLoRA` instance.
    """
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers


def count_parameters(model):
    """Return total/trainable/frozen parameter counts and trainable percentage."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


def merge_lora_weights(model):
    """Fold each `LinearWithLoRA`'s low-rank update into its base linear weight in place.

    After merging, every `LinearWithLoRA` submodule is replaced by its plain
    `nn.Linear`, removing the LoRA branch with no change to inference output.
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)


def quantize_to_nf4(tensor, block_size=64):
    """Quantize a tensor to 4-bit signed integers using per-block scaling.

    A simplified stand-in for QLoRA's NF4 quantization: the tensor is split
    into blocks, each scaled by its own max-abs value, then rounded into the
    signed 4-bit range [-8, 7].

    Returns:
        Tuple of (quantized int8 tensor, per-block scale tensor).
    """
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales


def dequantize_from_nf4(quantized, scales, original_shape):
    """Reconstruct a float tensor of `original_shape` from NF4 quantized blocks and scales."""
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)


def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    """Train only the trainable (LoRA) parameters of `model` with AdamW + MSE loss.

    Args:
        model: Model containing a mix of frozen and trainable parameters.
        data: Dict with "inputs" and "targets" tensors.
        epochs: Number of passes over the data.
        lr: Learning rate.
        batch_size: Mini-batch size.

    Returns:
        List of average loss per epoch.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses


def demo():
    """Run an end-to-end LoRA demo: inject, train, merge, and report parameter counts."""
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }


if __name__ == "__main__":
    results = demo()
    print(results)
