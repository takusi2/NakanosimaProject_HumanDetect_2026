"""GPU上での候補切り出し、色差スコア、分散フィルタ。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .features import scan_positions
from .types import Template


@dataclass(frozen=True)
class ScoreMap:
    """全配置の色差スコアと、詳細照合時の分散選別結果。"""

    scores: torch.Tensor
    x_positions: list[int]
    y_positions: list[int]
    variance_distances: torch.Tensor | None = None
    variance_passes: torch.Tensor | None = None
    flatness_passes: torch.Tensor | None = None
    relative_variance_passes: torch.Tensor | None = None
    candidate_variances: torch.Tensor | None = None


def score_map(
    frame_features: torch.Tensor,
    template: Template,
    stride: int,
    channel_weights: torch.Tensor,
    *,
    variance_channel_weights: torch.Tensor | None = None,
    variance_log_distance_max: float | None = None,
    variance_epsilon: float = 1.0,
    min_chroma_variance: float = 0.0,
    min_brightness_variance: float = 0.0,
) -> ScoreMap:
    """全配置候補の色差をGPUで求め、平坦・分散差の大きい候補を除外する。"""
    x_positions = scan_positions(frame_features.shape[1] - template.width, stride)
    y_positions = scan_positions(frame_features.shape[0] - template.height, stride)
    candidates = extract_candidate_patches(
        frame_features, x_positions, y_positions, template.width, template.height
    )
    if variance_channel_weights is None:
        difference = template.features.unsqueeze(0) - candidates
        pixel_error = torch.sqrt(torch.sum(difference.square() * channel_weights, dim=3))
        scores = torch.sum(pixel_error * template.weights, dim=(1, 2)) / template.weights.sum()
        return ScoreMap(scores.reshape(len(y_positions), len(x_positions)), x_positions, y_positions)

    candidate_variance = candidates.var(dim=(1, 2), correction=0)
    variance_distances = torch.sum(
        torch.abs(
            torch.log(candidate_variance + variance_epsilon)
            - torch.log(template.feature_variance + variance_epsilon)
        )
        * variance_channel_weights,
        dim=1,
    )
    relative_variance_passes = variance_distances <= variance_log_distance_max
    chroma_variance = candidate_variance[:, 0] + candidate_variance[:, 1]
    flatness_passes = (chroma_variance >= min_chroma_variance) | (
        candidate_variance[:, 2] >= min_brightness_variance
    )
    variance_passes = relative_variance_passes & flatness_passes
    difference = template.features.unsqueeze(0) - candidates
    pixel_error = torch.sqrt(torch.sum(difference.square() * channel_weights, dim=3))
    unfiltered_scores = torch.sum(pixel_error * template.weights, dim=(1, 2)) / template.weights.sum()
    scores = torch.where(variance_passes, unfiltered_scores, torch.full_like(unfiltered_scores, float("inf")))
    return ScoreMap(
        scores.reshape(len(y_positions), len(x_positions)),
        x_positions,
        y_positions,
        variance_distances.reshape(len(y_positions), len(x_positions)),
        variance_passes.reshape(len(y_positions), len(x_positions)),
        flatness_passes.reshape(len(y_positions), len(x_positions)),
        relative_variance_passes.reshape(len(y_positions), len(x_positions)),
        candidate_variance.reshape(len(y_positions), len(x_positions), 3),
    )


def extract_candidate_patches(
    frame_features: torch.Tensor,
    x_positions: list[int],
    y_positions: list[int],
    width: int,
    height: int,
) -> torch.Tensor:
    """全候補のROIパッチをGPUの高度インデックスで一括抽出する。"""
    x_starts = torch.as_tensor(x_positions, device=frame_features.device, dtype=torch.long)
    y_starts = torch.as_tensor(y_positions, device=frame_features.device, dtype=torch.long)
    x_indices = x_starts[:, None] + torch.arange(width, device=frame_features.device)
    y_indices = y_starts[:, None] + torch.arange(height, device=frame_features.device)
    patches = frame_features[y_indices[:, None, :, None], x_indices[None, :, None, :]]
    return patches.reshape(len(y_positions) * len(x_positions), height, width, 3)
