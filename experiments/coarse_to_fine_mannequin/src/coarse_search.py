"""縮小フレーム上の粗探索、NMS、詳細探索ROIの生成。"""

from __future__ import annotations

import numpy as np
import torch

from .scoring import score_map
from .types import Candidate, Roi, Template


def find_coarse_candidates(
    coarse_features: torch.Tensor,
    *,
    coarse_templates: list[Template],
    coarse_stride: int,
    coarse_candidates_per_template: int,
    coarse_top_k: int,
    frame_downscale: float,
    nms_distance_original_px: int,
    channel_weights: torch.Tensor,
    device: torch.device,
) -> list[Candidate]:
    """縮小フレーム全体を照合し、NMS後の上位候補だけを返す。"""
    candidate_rows: list[torch.Tensor] = []
    for template_index, template in enumerate(coarse_templates):
        if template.height > coarse_features.shape[0] or template.width > coarse_features.shape[1]:
            continue
        candidate_scores = score_map(coarse_features, template, coarse_stride, channel_weights)
        count = min(coarse_candidates_per_template, candidate_scores.scores.numel())
        values, indices = torch.topk(candidate_scores.scores.flatten(), k=count, largest=False)
        x_positions = torch.as_tensor(candidate_scores.x_positions, device=device, dtype=torch.long)
        y_positions = torch.as_tensor(candidate_scores.y_positions, device=device, dtype=torch.long)
        x = x_positions[torch.remainder(indices, len(candidate_scores.x_positions))]
        y = y_positions[torch.div(indices, len(candidate_scores.x_positions), rounding_mode="floor")]
        candidate_rows.append(
            torch.stack(
                (values, x.to(torch.float32), y.to(torch.float32), torch.full_like(values, template_index)),
                dim=1,
            )
        )

    if not candidate_rows:
        raise ValueError("no coarse template fits inside the downscaled frame")
    all_candidates = torch.cat(candidate_rows, dim=0)
    scores = all_candidates[:, 0]
    template_indices = all_candidates[:, 3].to(torch.long)
    template_widths = torch.as_tensor([item.width for item in coarse_templates], device=device, dtype=torch.float32)
    template_heights = torch.as_tensor([item.height for item in coarse_templates], device=device, dtype=torch.float32)
    center_x = (all_candidates[:, 1] + template_widths[template_indices] / 2) / frame_downscale
    center_y = (all_candidates[:, 2] + template_heights[template_indices] / 2) / frame_downscale

    available = torch.ones(scores.shape[0], dtype=torch.bool, device=device)
    selected_rows: list[torch.Tensor] = []
    for _ in range(min(coarse_top_k, scores.shape[0])):
        masked_scores = scores.masked_fill(~available, float("inf"))
        selected_index = torch.argmin(masked_scores)
        selected_rows.append(torch.cat((masked_scores[selected_index].reshape(1), all_candidates[selected_index, 1:])))
        available[selected_index] = False
        distance_squared = (center_x - center_x[selected_index]).square() + (center_y - center_y[selected_index]).square()
        available &= distance_squared >= nms_distance_original_px**2

    selected_values = torch.stack(selected_rows).detach().cpu().numpy()
    selected: list[Candidate] = []
    for score, x, y, template_index in selected_values:
        if not np.isfinite(score):
            continue
        selected.append(Candidate(coarse_templates[int(template_index)], int(x), int(y), float(score)))
    if not selected:
        raise ValueError("no coarse candidates remained after NMS")
    return selected


def make_rois(
    candidates: list[Candidate],
    frame_shape: tuple[int, int],
    *,
    detail_templates: list[Template],
    frame_downscale: float,
    coarse_stride: int,
    roi_margin_px: int,
) -> list[Roi]:
    """粗探索候補の中心を元解像度へ戻し、詳細探索用ROIを作る。"""
    frame_height, frame_width = frame_shape
    max_template_width = max(template.width for template in detail_templates)
    max_template_height = max(template.height for template in detail_templates)
    coarse_uncertainty = coarse_stride / frame_downscale
    roi_width = min(frame_width, round(max_template_width + 2 * (roi_margin_px + coarse_uncertainty)))
    roi_height = min(frame_height, round(max_template_height + 2 * (roi_margin_px + coarse_uncertainty)))
    rois: list[Roi] = []
    for index, candidate in enumerate(candidates, start=1):
        center_x = round((candidate.x + candidate.template.width / 2) / frame_downscale)
        center_y = round((candidate.y + candidate.template.height / 2) / frame_downscale)
        x = max(0, min(frame_width - roi_width, center_x - roi_width // 2))
        y = max(0, min(frame_height - roi_height, center_y - roi_height // 2))
        rois.append(Roi(index, x, y, roi_width, roi_height, candidate))
    return rois
