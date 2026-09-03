"""GPU上のBGR画像を二十反対色と輝度の特徴量へ変換する。"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def bgr_to_opponent_features(
    image_bgr: torch.Tensor,
    brightness_weights: Sequence[float] = (0.2126, 0.7152, 0.0722),
) -> torch.Tensor:
    """Return an ``(H, W, 3)`` tensor ordered as RG, BY, Y.

    ``image_bgr`` はすでに GPU に転送済みの float32 Tensor を受け取る。
    画素値は 0--255 のまま扱うため、スコアは 8 bit 色差として読める。
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")

    weights = torch.as_tensor(brightness_weights, dtype=torch.float32, device=image_bgr.device)
    if weights.shape != (3,) or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("brightness_weights must contain three non-negative values")
    weights = weights / weights.sum()

    blue = image_bgr[..., 0]
    green = image_bgr[..., 1]
    red = image_bgr[..., 2]

    rg = red - green
    by = (red + green) * 0.5 - blue
    brightness = weights[0] * red + weights[1] * green + weights[2] * blue
    return torch.stack((rg, by, brightness), dim=-1)
