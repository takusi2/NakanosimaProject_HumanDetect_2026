"""テンプレート画素の重要度を表す二次元重みマップ。"""

from __future__ import annotations

import numpy as np


def center_falloff(height: int, width: int, min_weight: float = 0.05) -> np.ndarray:
    """中心が 1、四隅が ``min_weight`` の滑らかな楕円重みを返す。"""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be between 0 and 1")

    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    normalized_radius = np.clip(radius / np.sqrt(2.0), 0.0, 1.0)
    return (min_weight + (1.0 - min_weight) * (1.0 - normalized_radius)).astype(np.float32)


def inner_rectangle(
    height: int,
    width: int,
    margins: tuple[float, float, float, float],
) -> np.ndarray:
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

    weights = np.zeros((height, width), dtype=np.float32)
    weights[y0:y1, x0:x1] = 1.0
    return weights
