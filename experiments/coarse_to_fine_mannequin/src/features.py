"""画像を照合用の反対色特徴量へ変換する処理。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def to_gpu_bgr(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image_bgr)).to(device=device, dtype=torch.float32)


def normalise_weights(values: Sequence[float], device: torch.device) -> torch.Tensor:
    weights = torch.as_tensor(values, dtype=torch.float32, device=device)
    if weights.shape != (3,) or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("brightness_weights must contain three non-negative values")
    return weights / weights.sum()


def normalise_channel_weights(values: Sequence[float], device: torch.device) -> torch.Tensor:
    weights = torch.as_tensor(values, dtype=torch.float32, device=device)
    if weights.shape != (3,) or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("channel_weights must contain three non-negative values")
    return weights


def bgr_to_features(image_bgr: torch.Tensor, brightness_weights: torch.Tensor) -> torch.Tensor:
    blue, green, red = image_bgr[..., 0], image_bgr[..., 1], image_bgr[..., 2]
    rg = red - green
    by = (red + green) * 0.5 - blue
    brightness = brightness_weights[0] * red + brightness_weights[1] * green + brightness_weights[2] * blue
    return torch.stack((rg, by, brightness), dim=-1)


def center_weight(height: int, width: int, min_weight: float, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    distance = torch.sqrt(y[:, None].square() + x[None, :].square()) / (2.0**0.5)
    return min_weight + (1.0 - min_weight) * (1.0 - torch.clamp(distance, 0.0, 1.0))


def downscale_on_gpu(image_bgr: torch.Tensor, scale: float) -> torch.Tensor:
    height = max(1, round(image_bgr.shape[0] * scale))
    width = max(1, round(image_bgr.shape[1] * scale))
    chw = image_bgr.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(chw, size=(height, width), mode="area").squeeze(0).permute(1, 2, 0)


def scan_positions(maximum: int, stride: int) -> list[int]:
    positions = list(range(0, maximum + 1, stride))
    if positions[-1] != maximum:
        positions.append(maximum)
    return positions
