import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CaSEBlock(nn.Module):
    """Minimal CaSE-style gating module per channel."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        hidden = max(4, channels // reduction)
        self.gate_net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))  # learnable per-channel scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Basic: modulate by per-channel gamma
        return x * self.gamma

    def compute_gamma(self) -> torch.Tensor:
        with torch.no_grad():
            return self.gamma.clone()


class CTGR_CABlock(nn.Module):
    """CaSE block with Cross-Task Gamma Regularization (EMA priors)."""
    def __init__(self, channels: int, lambda_reg: float = 0.0, eta: float = 0.1):
        super().__init__()
        self.cas_block = CaSEBlock(channels)
        self.lambda_reg = lambda_reg
        self.eta = eta  # EMA step
        self.register_buffer("gamma_mu", torch.ones(1, channels, 1, 1))
        self._batch_gamma_sum = None
        self._batch_gamma_count = 0
        self._reg_loss = torch.tensor(0.0, dtype=torch.float32)
        # alias for clarity
        self.ctgr_name = f"CTGRBlock-{channels}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cas_block(x)
        if self.training and self.lambda_reg > 0:
            gamma = self.cas_block.compute_gamma()
            gamma_mean = gamma.mean(dim=(0, 2, 3), keepdim=True)
            if self._batch_gamma_sum is None:
                self._batch_gamma_sum = gamma_mean.clone()
                self._batch_gamma_count = 1
            else:
                self._batch_gamma_sum += gamma_mean
                self._batch_gamma_count += 1
            self._reg_loss = self.lambda_reg * ((gamma_mean - self.gamma_mu) ** 2).sum()
        else:
            self._reg_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return y

    @property
    def reg_loss(self) -> torch.Tensor:
        return self._reg_loss

    def finalize_batch(self) -> None:
        if self._batch_gamma_count > 0:
            batch_mean = self._batch_gamma_sum / self._batch_gamma_count
            self.gamma_mu = (1.0 - self.eta) * self.gamma_mu + self.eta * batch_mean
        self._batch_gamma_sum = None
        self._batch_gamma_count = 0
        self._reg_loss = torch.tensor(0.0, dtype=torch.float32, device=self.gamma_mu.device)


class SimpleBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16, lambda_reg: float = 0.01, eta: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = CTGR_CABlock(base_channels, lambda_reg=lambda_reg, eta=eta)
        self.stage2 = CTGR_CABlock(base_channels, lambda_reg=lambda_reg, eta=eta)
        self.stage3 = CTGR_CABlock(base_channels, lambda_reg=lambda_reg, eta=eta)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = base_channels
        self.ctgr_blocks: List[CTGR_CABlock] = [self.stage1, self.stage2, self.stage3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class CTGRNet(nn.Module):
    def __init__(self, cfg, num_classes: int = 10) -> None:
        super().__init__()
        base_channels = int(cfg.model.get("base_channels", 16))
        lambda_reg = float(cfg.training.get("reg_lambda", 0.01))
        eta = float(cfg.training.get("eta", 0.1))
        self.backbone = SimpleBackbone(in_channels=3, base_channels=base_channels, lambda_reg=lambda_reg, eta=eta)
        self.fc = nn.Linear(self.backbone.out_dim, num_classes)
        self.ctgr_blocks = self.backbone.ctgr_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.fc(feats)
        return logits


class CaSEBaselineNet(nn.Module):
    def __init__(self, cfg, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ca = CaSEBlock(16)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.ca(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


__all__ = ["CTGRNet", "CaSEBaselineNet", "CaSEBlock", "CTGR_CABlock"]

# Re-export simple gate block for external usage if needed
class CaSEBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
    def compute_gamma(self) -> torch.Tensor:
        with torch.no_grad():
            return self.gamma.clone()
