"""二十反対色の重み付きスライディングウィンドウ照合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from .opponent_color import bgr_to_opponent_features
from .weight_map import center_falloff, inner_rectangle

WeightMode = Literal["center_falloff", "inner_rectangle"]


@dataclass(frozen=True)
class MatchResult:
    """1 フレームの照合結果。座標は入力フレームにおけるテンプレート左上。"""

    top_left: tuple[int, int]
    center: tuple[float, float]
    score: float
    detected: bool
    score_map: torch.Tensor


class OpponentColorMatcher:
    """テンプレートをGPUへ一度だけ転送し、各フレームをGPUで照合する。"""

    def __init__(
        self,
        template_bgr: np.ndarray,
        *,
        max_error: float,
        brightness_weights: Sequence[float] = (0.2126, 0.7152, 0.0722),
        channel_weights: Sequence[float] = (1.0, 1.0, 1.0),
        weight_mode: WeightMode = "inner_rectangle",
        min_weight: float = 0.05,
        margins: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 0.15),
        device: str = "cuda",
    ) -> None:
        if max_error < 0:
            raise ValueError("max_error must be non-negative")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA が利用できません。NVIDIA GPU と CUDA 版 PyTorch を確認してください。")

        template_tensor = torch.from_numpy(np.ascontiguousarray(template_bgr)).to(
            device=self.device, dtype=torch.float32
        )
        self.template_features = bgr_to_opponent_features(template_tensor, brightness_weights)
        self.template_height, self.template_width = self.template_features.shape[:2]
        self.max_error = float(max_error)

        channels = torch.as_tensor(channel_weights, dtype=torch.float32, device=self.device)
        if channels.shape != (3,) or torch.any(channels < 0) or float(channels.sum()) <= 0:
            raise ValueError("channel_weights must contain three non-negative values")
        self.channel_weights = channels

        if weight_mode == "center_falloff":
            self.template_weights = center_falloff(
                self.template_height, self.template_width, min_weight, device=self.device
            )
        elif weight_mode == "inner_rectangle":
            self.template_weights = inner_rectangle(
                self.template_height, self.template_width, margins, device=self.device
            )
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode}")
        self._weight_sum = float(self.template_weights.sum())

    def match(
        self,
        frame_bgr: np.ndarray,
        *,
        stride_x: int | None = None,
        stride_y: int | None = None,
    ) -> MatchResult:
        """Search a frame and return its minimum-error template placement.

        ``score_map[row, column]`` corresponds to the positions returned by
        the respective y/x scan coordinates.  The last possible x/y position
        is always evaluated, even when it is not an exact stride multiple.
        """
        with torch.inference_mode():
            # OpenCVが返すCPUフレームをGPUへ転送してから、色変換・誤差計算をGPUで行う。
            frame_tensor = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(
                device=self.device, dtype=torch.float32
            )
            features = bgr_to_opponent_features(frame_tensor)
            frame_height, frame_width = features.shape[:2]
            if frame_height < self.template_height or frame_width < self.template_width:
                raise ValueError("frame must be at least as large as the template")

            sx = stride_x if stride_x is not None else max(1, self.template_width // 4)
            sy = stride_y if stride_y is not None else max(1, self.template_height // 4)
            if sx <= 0 or sy <= 0:
                raise ValueError("stride_x and stride_y must be positive")

            x_positions = _scan_positions(frame_width - self.template_width, sx)
            y_positions = _scan_positions(frame_height - self.template_height, sy)

            # 全候補をGPU上に並べ、差分・平方根・重み付き平均をまとめてGPUで計算する。
            candidates = torch.stack(
                [
                    features[y : y + self.template_height, x : x + self.template_width]
                    for y in y_positions
                    for x in x_positions
                ]
            )
            feature_delta = self.template_features.unsqueeze(0) - candidates
            pixel_error = torch.sqrt(
                torch.sum(feature_delta.square() * self.channel_weights, dim=3)
            )
            scores = torch.sum(pixel_error * self.template_weights, dim=(1, 2)) / self._weight_sum
            score_map = scores.reshape(len(y_positions), len(x_positions))

            best_index = int(torch.argmin(scores).item())
            best_y = y_positions[best_index // len(x_positions)]
            best_x = x_positions[best_index % len(x_positions)]
            best_score = float(scores[best_index].item())

        return MatchResult(
            top_left=(best_x, best_y),
            center=(best_x + self.template_width / 2, best_y + self.template_height / 2),
            score=best_score,
            detected=best_score <= self.max_error,
            score_map=score_map,
        )


def _scan_positions(maximum: int, stride: int) -> list[int]:
    positions = list(range(0, maximum + 1, stride))
    if positions[-1] != maximum:
        positions.append(maximum)
    return positions
