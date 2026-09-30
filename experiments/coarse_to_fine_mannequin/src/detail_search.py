"""元解像度ROI内の詳細探索と分散フィルタ。"""

from __future__ import annotations

import numpy as np
import torch

from .scoring import ScoreMap, score_map
from .types import DetailMatch, DetailVarianceFilter, Roi, Template


def find_detail_matches(
    original_features: torch.Tensor,
    rois: list[Roi],
    *,
    detail_templates: list[Template],
    detail_stride_for_scale,
    channel_weights: torch.Tensor,
    variance_channel_weights: torch.Tensor,
    variance_log_distance_max: float,
    variance_epsilon: float,
    min_chroma_variance: float,
    min_brightness_variance: float,
    collect_variance_filters: bool,
) -> tuple[list[DetailMatch], list[DetailVarianceFilter]]:
    """各ROI・各テンプレート倍率を照合し、倍率ごとの最良候補を返す。"""
    variance_filters: list[DetailVarianceFilter] = []
    pending_matches: list[tuple[Roi, Template, int, ScoreMap, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for roi in rois:
        roi_features = original_features[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        for template in detail_templates:
            if template.height > roi.height or template.width > roi.width:
                continue
            stride = detail_stride_for_scale(template.scale)
            candidate_scores = score_map(
                roi_features,
                template,
                stride,
                channel_weights,
                variance_channel_weights=variance_channel_weights,
                variance_log_distance_max=variance_log_distance_max,
                variance_epsilon=variance_epsilon,
                min_chroma_variance=min_chroma_variance,
                min_brightness_variance=min_brightness_variance,
            )
            if (
                candidate_scores.variance_passes is None
                or candidate_scores.flatness_passes is None
                or candidate_scores.relative_variance_passes is None
                or candidate_scores.candidate_variances is None
            ):
                raise RuntimeError("detail score map must include variance filter results")
            if collect_variance_filters:
                variance_filters.append(
                    DetailVarianceFilter(
                        roi=roi,
                        template=template,
                        x_positions=candidate_scores.x_positions,
                        y_positions=candidate_scores.y_positions,
                        variance_passes=candidate_scores.variance_passes.detach().cpu().numpy(),
                        flatness_passes=candidate_scores.flatness_passes.detach().cpu().numpy(),
                        relative_variance_passes=candidate_scores.relative_variance_passes.detach().cpu().numpy(),
                        candidate_variances=candidate_scores.candidate_variances.detach().cpu().numpy(),
                        min_chroma_variance=min_chroma_variance,
                        min_brightness_variance=min_brightness_variance,
                        stride=stride,
                    )
                )
            flat_scores = candidate_scores.scores.flatten()
            best_index = torch.argmin(flat_scores)
            pending_matches.append(
                (
                    roi,
                    template,
                    stride,
                    candidate_scores,
                    best_index,
                    flat_scores[best_index],
                    candidate_scores.variance_distances.flatten()[best_index],
                )
            )

    if not pending_matches:
        return [], variance_filters
    compact_results = torch.stack(
        [torch.stack((item[4].to(torch.float32), item[5], item[6])) for item in pending_matches]
    ).detach().cpu().numpy()
    matches: list[DetailMatch] = []
    for (roi, template, stride, candidate_scores, _, _, _), (best_index, score, variance_distance) in zip(
        pending_matches, compact_results
    ):
        if not np.isfinite(score):
            continue
        index = int(best_index)
        local_x = candidate_scores.x_positions[index % len(candidate_scores.x_positions)]
        local_y = candidate_scores.y_positions[index // len(candidate_scores.x_positions)]
        matches.append(
            DetailMatch(
                roi,
                template,
                roi.x + local_x,
                roi.y + local_y,
                float(score),
                float(variance_distance),
                stride,
            )
        )
    return matches, variance_filters
