import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      # nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(2)                       # AvgPool2d(kernel_size)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)            # Linear(in_features, out_features)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: (N, 1, 32, 32) 
        x = self.pool(torch.tanh(self.conv1(x)))   # (N, 6, 28, 28) -> (N, 6, 14, 14)
        x = self.pool(torch.tanh(self.conv2(x)))   # (N, 16, 10, 10) -> (N, 16, 5, 5)
        x = torch.flatten(x, 1)                    # (N, 400)
        x = torch.tanh(self.fc1(x))               # (N, 120)
        x = torch.tanh(self.fc2(x))               # (N, 84)
        return self.fc3(x)                         # (N, num_classes)

def LeNet5_implement():
    net = LeNet5()
    x = torch.randn(1, 1, 32, 32)
    print(f"output: {net(x).shape}")
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")

class VGGBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)  # padding=1 保持 H,W 不變
        self.bn1 = nn.BatchNorm2d(out_c)                                # (num_features) 對 channel 維度做 BN
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2)                                     # (kernel_size) H,W 各縮小一半

    def forward(self, x):
        # x: (N, in_c, H, W)
        x = F.relu(self.bn1(self.conv1(x)))   # (N, out_c, H, W)
        x = F.relu(self.bn2(self.conv2(x)))   # (N, out_c, H, W)
        return self.pool(x)                    # (N, out_c, H//2, W//2)
    
class VGG3LBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)  # padding=1 保持 H,W 不變
        self.bn1 = nn.BatchNorm2d(out_c)                                # (num_features) 對 channel 維度做 BN
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2)                                     # (kernel_size) H,W 各縮小一半

    def forward(self, x):
        # x: (N, in_c, H, W)
        x = F.relu(self.bn1(self.conv1(x)))   # (N, out_c, H, W)
        x = F.relu(self.bn2(self.conv2(x)))   # (N, out_c, H, W)
        x = F.relu(self.bn3(self.conv3(x)))   # (N, out_c, H, W)
        return self.pool(x)                    # (N, out_c, H//2, W//2)    

class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stack = nn.Sequential(
            VGGBlock(3, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (output_size) 不管輸入 H,W，輸出固定為 1x1
            nn.Flatten(),               # (N, C, 1, 1) -> (N, C)
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x:          (N, 3, 32, 32)
        # stack:      (N, 32, 16, 16) -> (N, 64, 8, 8) -> (N, 128, 4, 4)
        # AvgPool+Flatten: (N, 128)
        # Linear:     (N, num_classes)
        return self.head(self.stack(x))

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stack = nn.Sequential(
            VGGBlock(3, 64),          # (N, 3, 224, 224)  -> (N, 64, 112, 112)
            VGGBlock(64, 128),        # (N, 64, 112, 112) -> (N, 128, 56, 56)
            VGG3LBlock(128, 256),     # (N, 128, 56, 56)  -> (N, 256, 28, 28)
            VGG3LBlock(256, 512),     # (N, 256, 28, 28)  -> (N, 512, 14, 14)
            VGG3LBlock(512, 512),     # (N, 512, 14, 14)  -> (N, 512, 7, 7)
        )
        self.head = nn.Sequential(
            nn.Flatten(),                      # (N, 512, 7, 7) -> (N, 25088)
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),                   # (p) 訓練時隨機關閉 50% 神經元，防止 overfitting
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # x:      (N, 3, 224, 224)
        # stack:  (N, 64, 112, 112) -> (N, 128, 56, 56) -> (N, 256, 28, 28) -> (N, 512, 14, 14) -> (N, 512, 7, 7)
        # head:   (N, 25088) -> (N, 4096) -> (N, 4096) -> (N, num_classes)
        return self.head(self.stack(x))

def MiniVGG_implement():
    net = MiniVGG()
    x = torch.randn(1, 3, 32, 32)
    print(f"output: {net(x).shape}")
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")

def VGG16_implement():
    net = VGG16()
    x = torch.randn(1, 3, 224, 224)
    print(f"output: {net(x).shape}")
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_group(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_group(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_group(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_group(128, 256, num_blocks=2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def _make_group(self, in_c, out_c, num_blocks, stride):
        blocks = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

def TinyResNet_implement():
    net = TinyResNet()
    x = torch.randn(1, 3, 32, 32)
    print(f"output: {net(x).shape}")
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")


if __name__ == "__main__":
    # LeNet5_implement()
    # MiniVGG_implement()
    # VGG16_implement()
    TinyResNet_implement()
