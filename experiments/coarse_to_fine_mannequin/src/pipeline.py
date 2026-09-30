"""粗探索・詳細探索を順番に呼び出す、検出器の窓口。"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch

from .coarse_search import find_coarse_candidates, make_rois
from .detail_search import find_detail_matches
from .features import (
    bgr_to_features,
    center_weight,
    downscale_on_gpu,
    normalise_channel_weights,
    normalise_weights,
    to_gpu_bgr,
)
from .types import Candidate, DetailMatch, DetailVarianceFilter, FrameResult, Roi, Template

__all__ = [
    "Candidate",
    "CoarseToFineMatcher",
    "DetailMatch",
    "DetailVarianceFilter",
    "FrameResult",
    "Roi",
    "Template",
]


class CoarseToFineMatcher:
    """縮小フレームで候補を絞り、元解像度ROIだけを詳細照合する。"""

    def __init__(
        self,
        template_bgr: np.ndarray,
        *,
        device: str = "cuda",
        frame_downscale: float = 0.25,
        coarse_template_scales: Sequence[float] = (0.2,),
        detail_template_scales: Sequence[float] = (1.0, 0.8, 0.6, 0.4),
        coarse_stride: int = 2,
        detail_stride_base: int = 4,
        detail_stride_min: int = 2,
        coarse_top_k: int = 3,
        coarse_candidates_per_template: int = 20,
        nms_distance_original_px: int = 120,
        roi_margin_px: int = 32,
        final_max_error: float = 30.0,
        brightness_weights: Sequence[float] = (0.299, 0.587, 0.114),
        channel_weights: Sequence[float] = (0.5, 0.5, 1.0),
        variance_channel_weights: Sequence[float] = (1.0, 1.0, 1.0),
        variance_log_distance_max: float = 3.0,
        variance_epsilon: float = 1.0,
        min_chroma_variance: float = 0.0,
        min_brightness_variance: float = 0.0,
        min_weight: float = 0.05,
    ) -> None:
        if not 0.0 < frame_downscale < 1.0:
            raise ValueError("frame_downscale must be in the range (0, 1)")
        if coarse_stride <= 0 or detail_stride_base <= 0 or detail_stride_min <= 0:
            raise ValueError("strides must be positive")
        if detail_stride_min > detail_stride_base:
            raise ValueError("detail_stride_min must not exceed detail_stride_base")
        if coarse_top_k <= 0 or coarse_candidates_per_template <= 0:
            raise ValueError("candidate counts must be positive")
        if variance_log_distance_max < 0 or variance_epsilon <= 0:
            raise ValueError("variance thresholds must be positive")
        if min_chroma_variance < 0 or min_brightness_variance < 0:
            raise ValueError("minimum variances must be non-negative")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA GPU and CUDA PyTorch installation.")
        self.frame_downscale = frame_downscale
        self.coarse_stride = coarse_stride
        self.detail_stride_base = detail_stride_base
        self.detail_stride_min = detail_stride_min
        self.coarse_top_k = coarse_top_k
        self.coarse_candidates_per_template = coarse_candidates_per_template
        self.nms_distance_original_px = nms_distance_original_px
        self.roi_margin_px = roi_margin_px
        self.final_max_error = final_max_error
        self.min_weight = min_weight
        self.brightness_weights = normalise_weights(brightness_weights, self.device)
        self.channel_weights = normalise_channel_weights(channel_weights, self.device)
        self.variance_channel_weights = normalise_channel_weights(variance_channel_weights, self.device)
        self.variance_log_distance_max = variance_log_distance_max
        self.variance_epsilon = variance_epsilon
        self.min_chroma_variance = min_chroma_variance
        self.min_brightness_variance = min_brightness_variance

        self.coarse_templates = self._create_templates(template_bgr, coarse_template_scales, "coarse")
        self.detail_templates = self._create_templates(template_bgr, detail_template_scales, "detail")
        if not self.coarse_templates or not self.detail_templates:
            raise ValueError("at least one coarse and one detail template are required")

    def detail_stride_for_scale(self, scale: float) -> int:
        """小テンプレートほど細かく探索するためのstrideを返す。"""
        return max(self.detail_stride_min, int(self.detail_stride_base * scale))

    def _create_templates(
        self, original_bgr: np.ndarray, scales: Sequence[float], prefix: str
    ) -> list[Template]:
        templates: list[Template] = []
        used_sizes: set[tuple[int, int]] = set()
        for scale in scales:
            scale = float(scale)
            if not 0.0 < scale <= 1.0:
                raise ValueError("template scales must be in the range (0, 1]")
            width = max(1, round(original_bgr.shape[1] * scale))
            height = max(1, round(original_bgr.shape[0] * scale))
            if (width, height) in used_sizes:
                continue
            used_sizes.add((width, height))
            image = cv2.resize(original_bgr, (width, height), interpolation=cv2.INTER_AREA)
            tensor = to_gpu_bgr(image, self.device)
            features = bgr_to_features(tensor, self.brightness_weights)
            weights = center_weight(height, width, self.min_weight, self.device)
            feature_variance = features.var(dim=(0, 1), correction=0)
            templates.append(Template(f"{prefix}_{scale:.3f}", scale, image, features, weights, feature_variance))
        return templates

    def process(
        self,
        frame_bgr: np.ndarray,
        *,
        collect_coarse_image: bool = True,
        collect_detail_variance_filters: bool = True,
    ) -> FrameResult:
        """1フレームを粗探索、ROI生成、詳細探索の順で照合する。"""
        with torch.inference_mode():
            original_bgr_gpu = to_gpu_bgr(frame_bgr, self.device)
            original_features = bgr_to_features(original_bgr_gpu, self.brightness_weights)
            coarse_bgr_gpu = downscale_on_gpu(original_bgr_gpu, self.frame_downscale)
            coarse_features = bgr_to_features(coarse_bgr_gpu, self.brightness_weights)
            coarse_image_bgr = None
            if collect_coarse_image:
                coarse_image_bgr = coarse_bgr_gpu.round().to(torch.uint8).detach().cpu().numpy()

            coarse_candidates = find_coarse_candidates(
                coarse_features,
                coarse_templates=self.coarse_templates,
                coarse_stride=self.coarse_stride,
                coarse_candidates_per_template=self.coarse_candidates_per_template,
                coarse_top_k=self.coarse_top_k,
                frame_downscale=self.frame_downscale,
                nms_distance_original_px=self.nms_distance_original_px,
                channel_weights=self.channel_weights,
                device=self.device,
            )
            rois = make_rois(
                coarse_candidates,
                frame_bgr.shape[:2],
                detail_templates=self.detail_templates,
                frame_downscale=self.frame_downscale,
                coarse_stride=self.coarse_stride,
                roi_margin_px=self.roi_margin_px,
            )
            detail_matches, detail_variance_filters = find_detail_matches(
                original_features,
                rois,
                detail_templates=self.detail_templates,
                detail_stride_for_scale=self.detail_stride_for_scale,
                channel_weights=self.channel_weights,
                variance_channel_weights=self.variance_channel_weights,
                variance_log_distance_max=self.variance_log_distance_max,
                variance_epsilon=self.variance_epsilon,
                min_chroma_variance=self.min_chroma_variance,
                min_brightness_variance=self.min_brightness_variance,
                collect_variance_filters=collect_detail_variance_filters,
            )
            best_match = min(detail_matches, key=lambda item: item.score) if detail_matches else None
            return FrameResult(
                coarse_image_bgr=coarse_image_bgr,
                coarse_candidates=coarse_candidates,
                rois=rois,
                detail_matches=detail_matches,
                detail_variance_filters=detail_variance_filters,
                best_match=best_match,
                detected=best_match is not None and best_match.score <= self.final_max_error,
            )
