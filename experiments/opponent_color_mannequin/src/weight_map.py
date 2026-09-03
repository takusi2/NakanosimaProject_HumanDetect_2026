"""GPU上でテンプレート画素の重要度を表す二次元重みマップを作る。"""

from __future__ import annotations

import torch


def center_falloff(
    height: int, width: int, min_weight: float = 0.05, *, device: torch.device
) -> torch.Tensor:
    """中心が 1、四隅が ``min_weight`` の滑らかな楕円重みを返す。"""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be between 0 and 1")

    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
    radius = torch.sqrt(y[:, None].square() + x[None, :].square())
    normalized_radius = torch.clamp(radius / (2.0**0.5), 0.0, 1.0)
    return min_weight + (1.0 - min_weight) * (1.0 - normalized_radius)


def inner_rectangle(
    height: int,
    width: int,
    margins: tuple[float, float, float, float],
    *,
    device: torch.device,
) -> torch.Tensor:
    """指定した上・下・左・右の比率をゼロ重みとした矩形マップを返す。"""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    top, bottom, left, right = margins
    if any(not 0.0 <= margin < 0.5 for margin in margins):
        raise ValueError("each margin must be in the range [0, 0.5)")

    y0, y1 = int(round(height * top)), int(round(height * (1.0 - bottom)))
    x0, x1 = int(round(width * left)), int(round(width * (1.0 - right)))
    if y0 >= y1 or x0 >= x1:
        raise ValueError("margins leave no weighted pixels")

    weights = torch.zeros((height, width), dtype=torch.float32, device=device)
    weights[y0:y1, x0:x1] = 1.0
    return weights
