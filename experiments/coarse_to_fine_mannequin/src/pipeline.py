"""粗探索・詳細探索を順番に呼び出す、検出器の窓口。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

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

if TYPE_CHECKING:
    from .settings import MatcherSettings

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
        settings: MatcherSettings | None = None,
        **legacy_settings: object,
    ) -> None:
        """参照画像と型付き設定から照合器を作る。"""
        if settings is None:
            # 既存の直接引数呼出しを、テストと過去の利用コードのために維持する。
            # run.py は必ず MatcherSettings を渡す。
            from .settings import MatcherSettings

            settings = MatcherSettings(legacy_settings)
        elif legacy_settings:
            raise TypeError("settings と個別の設定値を同時に指定することはできません")
        device = settings.device
        frame_downscale = settings.frame_downscale
        coarse_template_scales = settings.coarse_template_scales
        detail_template_scales = settings.detail_template_scales
        coarse_stride = settings.coarse_stride
        detail_stride_base = settings.detail_stride_base
        detail_stride_min = settings.detail_stride_min
        coarse_top_k = settings.coarse_top_k
        coarse_candidates_per_template = settings.coarse_candidates_per_template
        nms_distance_original_px = settings.nms_distance_original_px
        roi_margin_px = settings.roi_margin_px
        final_max_error = settings.final_max_error
        brightness_weights = settings.brightness_weights
        channel_weights = settings.channel_weights
        variance_channel_weights = settings.variance_channel_weights
        variance_log_distance_max = settings.variance_log_distance_max
        variance_epsilon = settings.variance_epsilon
        min_chroma_variance = settings.min_chroma_variance
        min_brightness_variance = settings.min_brightness_variance
        min_weight = settings.min_weight
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
        """詳細探索でのスケールごとのstrideを返す。"""
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
            # 原画像と縮小画像の特徴量を計算してGPUに転送
            original_bgr_gpu = to_gpu_bgr(frame_bgr, self.device)
            original_features = bgr_to_features(original_bgr_gpu, self.brightness_weights)
            coarse_bgr_gpu = downscale_on_gpu(original_bgr_gpu, self.frame_downscale)
            coarse_features = bgr_to_features(coarse_bgr_gpu, self.brightness_weights)
            coarse_image_bgr = None
            if collect_coarse_image:
                coarse_image_bgr = coarse_bgr_gpu.round().to(torch.uint8).detach().cpu().numpy()

            # 粗探索
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
            # 粗探索での候補から詳細探索用ROIを生成
            rois = make_rois(
                coarse_candidates,
                frame_bgr.shape[:2],
                detail_templates=self.detail_templates,
                frame_downscale=self.frame_downscale,
                coarse_stride=self.coarse_stride,
                roi_margin_px=self.roi_margin_px,
            )
            # 詳細探索
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
