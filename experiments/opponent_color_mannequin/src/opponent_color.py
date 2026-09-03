"""BGR 画像を二十反対色と輝度の特徴量へ変換する。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bgr_to_opponent_features(
    image_bgr: np.ndarray,
    brightness_weights: Sequence[float] = (0.2126, 0.7152, 0.0722),
) -> np.ndarray:
    """Return an ``(H, W, 3)`` float32 array ordered as RG, BY, Y.

    Input pixels remain in the 0--255 scale.  Keeping that scale makes the
    resulting match score directly interpretable as an 8-bit colour distance.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")

    weights = np.asarray(brightness_weights, dtype=np.float32)
    if weights.shape != (3,) or np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("brightness_weights must contain three non-negative values")
    weights = weights / weights.sum()

    blue = image_bgr[..., 0].astype(np.float32)
    green = image_bgr[..., 1].astype(np.float32)
    red = image_bgr[..., 2].astype(np.float32)

    rg = red - green
    by = (red + green) * 0.5 - blue
    brightness = weights[0] * red + weights[1] * green + weights[2] * blue
    return np.stack((rg, by, brightness), axis=-1)
