"""GPU上で粗探索と詳細照合を行う二段階テンプレート照合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Template:
    """GPU上に保存した1種類の参照画像特徴量。"""

    name: str
    scale: float
    image_bgr: np.ndarray
    features: torch.Tensor
    weights: torch.Tensor
    feature_variance: torch.Tensor

    @property
    def height(self) -> int:
        return int(self.features.shape[0])

    @property
    def width(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class Candidate:
    """粗探索で選んだ候補。座標は縮小フレーム上のテンプレート左上。"""

    template: Template
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class Roi:
    """元解像度フレーム上で詳細照合する矩形。"""

    index: int
    x: int
    y: int
    width: int
    height: int
    coarse_candidate: Candidate


@dataclass(frozen=True)
class DetailMatch:
    """1つのROIと1つの詳細テンプレートの照合結果。座標は元フレーム上。"""

    roi: Roi
    template: Template
    x: int
    y: int
    score: float
    variance_distance: float
    stride: int


@dataclass(frozen=True)
class DetailVarianceFilter:
    """ROI・テンプレートごとの分散フィルタ判定。座標はROI内の左上位置。"""

    roi: Roi
    template: Template
    x_positions: list[int]
    y_positions: list[int]
    variance_passes: np.ndarray
    flatness_passes: np.ndarray
    relative_variance_passes: np.ndarray
    candidate_variances: np.ndarray
    min_chroma_variance: float
    min_brightness_variance: float
    stride: int

    @property
    def candidate_count(self) -> int:
        return int(self.variance_passes.size)

    @property
    def passed_count(self) -> int:
        return int(np.count_nonzero(self.variance_passes))

    @property
    def rejected_count(self) -> int:
        return self.candidate_count - self.passed_count

    @property
    def flatness_rejected_count(self) -> int:
        """色・明度ともに変化の少ない平坦領域として除外した候補数。"""
        return int(np.count_nonzero(~self.flatness_passes))

    @property
    def relative_variance_rejected_count(self) -> int:
        """平坦ではないが、参照との相対分散差で除外した候補数。"""
        return int(
            np.count_nonzero(self.flatness_passes & ~self.relative_variance_passes)
        )

    @property
    def candidate_variance_mean(self) -> np.ndarray:
        """候補全体におけるRG/BY/Brightness分散の平均。"""
        return self.candidate_variances.mean(axis=(0, 1))


@dataclass(frozen=True)
class FrameResult:
    """1フレームの粗探索・詳細照合の全結果。"""

    # 保存・可視化が必要な場合だけGPUからCPUへ取り出す。
    coarse_image_bgr: np.ndarray | None
    coarse_candidates: list[Candidate]
    rois: list[Roi]
    detail_matches: list[DetailMatch]
    detail_variance_filters: list[DetailVarianceFilter]
    best_match: DetailMatch | None
    detected: bool
    # run.py が時系列確認を行った場合にのみ設定する。Noneなら生の照合結果を示す。
    temporal_status: str | None = None
    track_id: int | None = None
    positive_frames: int | None = None
    history_window_frames: int | None = None


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
        self.brightness_weights = _normalise_weights(brightness_weights, self.device)
        self.channel_weights = _normalise_channel_weights(channel_weights, self.device)
        self.variance_channel_weights = _normalise_channel_weights(
            variance_channel_weights, self.device
        )
        self.variance_log_distance_max = variance_log_distance_max
        self.variance_epsilon = variance_epsilon
        self.min_chroma_variance = min_chroma_variance
        self.min_brightness_variance = min_brightness_variance

        self.coarse_templates = self._create_templates(template_bgr, coarse_template_scales, "coarse")
        self.detail_templates = self._create_templates(template_bgr, detail_template_scales, "detail")
        if not self.coarse_templates or not self.detail_templates:
            raise ValueError("at least one coarse and one detail template are required")

    def detail_stride_for_scale(self, scale: float) -> int:
        """詳細テンプレート倍率に対応するstrideを返す。

        小さなテンプレートほど細かく走査するため、
        ``max(detail_stride_min, floor(detail_stride_base * scale))`` を使う。
        """
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
            tensor = _to_gpu_bgr(image, self.device)
            features = _bgr_to_features(tensor, self.brightness_weights)
            weights = _center_weight(height, width, self.min_weight, self.device)
            feature_variance = features.var(dim=(0, 1), correction=0)
            templates.append(
                Template(f"{prefix}_{scale:.3f}", scale, image, features, weights, feature_variance)
            )
        return templates

    def process(
        self,
        frame_bgr: np.ndarray,
        *,
        collect_coarse_image: bool = True,
        collect_detail_variance_filters: bool = True,
    ) -> FrameResult:
        """1フレームを照合する。

        ``collect_*`` は保存・可視化用の中間データだけを制御する。検出結果は
        変えず、不要なGPU→CPU転送と全候補配列の保持を避ける。
        """
        with torch.inference_mode():
            # 元フレームは1回だけGPUへ転送する。詳細照合はこの特徴量のROIを切り出す。
            original_bgr_gpu = _to_gpu_bgr(frame_bgr, self.device)
            original_features = _bgr_to_features(original_bgr_gpu, self.brightness_weights)

            coarse_bgr_gpu = _downscale_on_gpu(original_bgr_gpu, self.frame_downscale)
            coarse_features = _bgr_to_features(coarse_bgr_gpu, self.brightness_weights)
            coarse_image_bgr = None
            if collect_coarse_image:
                coarse_image_bgr = (
                    coarse_bgr_gpu.round().to(torch.uint8).detach().cpu().numpy()
                )

            coarse_candidates = self._find_coarse_candidates(coarse_features)
            rois = self._make_rois(coarse_candidates, frame_bgr.shape[:2])
            detail_matches, detail_variance_filters = self._find_detail_matches(
                original_features, rois, collect_variance_filters=collect_detail_variance_filters
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

    def _find_coarse_candidates(self, coarse_features: torch.Tensor) -> list[Candidate]:
        candidate_rows: list[torch.Tensor] = []
        for template_index, template in enumerate(self.coarse_templates):
            if template.height > coarse_features.shape[0] or template.width > coarse_features.shape[1]:
                continue
            score_map = _score_map(
                coarse_features, template, self.coarse_stride, self.channel_weights
            )
            count = min(self.coarse_candidates_per_template, score_map.scores.numel())
            values, indices = torch.topk(score_map.scores.flatten(), k=count, largest=False)
            x_positions = torch.as_tensor(
                score_map.x_positions, device=self.device, dtype=torch.long
            )
            y_positions = torch.as_tensor(
                score_map.y_positions, device=self.device, dtype=torch.long
            )
            x = x_positions[torch.remainder(indices, len(score_map.x_positions))]
            y = y_positions[torch.div(indices, len(score_map.x_positions), rounding_mode="floor")]
            candidate_rows.append(
                torch.stack(
                    (
                        values,
                        x.to(torch.float32),
                        y.to(torch.float32),
                        torch.full_like(values, template_index),
                    ),
                    dim=1,
                )
            )

        if not candidate_rows:
            raise ValueError("no coarse template fits inside the downscaled frame")
        all_candidates = torch.cat(candidate_rows, dim=0)
        scores = all_candidates[:, 0]
        template_indices = all_candidates[:, 3].to(torch.long)
        template_widths = torch.as_tensor(
            [template.width for template in self.coarse_templates],
            device=self.device,
            dtype=torch.float32,
        )
        template_heights = torch.as_tensor(
            [template.height for template in self.coarse_templates],
            device=self.device,
            dtype=torch.float32,
        )
        center_x = (all_candidates[:, 1] + template_widths[template_indices] / 2) / self.frame_downscale
        center_y = (all_candidates[:, 2] + template_heights[template_indices] / 2) / self.frame_downscale

        # NMSもGPU上で行う。最後に選ばれた最大top_k件だけを一括でCPUへ渡す。
        available = torch.ones(scores.shape[0], dtype=torch.bool, device=self.device)
        selected_rows: list[torch.Tensor] = []
        for _ in range(min(self.coarse_top_k, scores.shape[0])):
            masked_scores = scores.masked_fill(~available, float("inf"))
            selected_index = torch.argmin(masked_scores)
            selected_rows.append(
                torch.cat(
                    (masked_scores[selected_index].reshape(1), all_candidates[selected_index, 1:])
                )
            )
            available[selected_index] = False
            distance_squared = (center_x - center_x[selected_index]).square() + (
                center_y - center_y[selected_index]
            ).square()
            available &= distance_squared >= self.nms_distance_original_px**2

        # FrameResultはPythonの描画・保存コードへ返すため、この小さな配列だけは境界で転送する。
        selected_values = torch.stack(selected_rows).detach().cpu().numpy()
        selected: list[Candidate] = []
        for score, x, y, template_index in selected_values:
            if not np.isfinite(score):
                continue
            selected.append(
                Candidate(
                    self.coarse_templates[int(template_index)], int(x), int(y), float(score)
                )
            )
        if not selected:
            raise ValueError("no coarse candidates remained after NMS")
        return selected

    def _make_rois(self, candidates: list[Candidate], frame_shape: tuple[int, int]) -> list[Roi]:
        frame_height, frame_width = frame_shape
        max_template_width = max(template.width for template in self.detail_templates)
        max_template_height = max(template.height for template in self.detail_templates)
        coarse_uncertainty = self.coarse_stride / self.frame_downscale
        roi_width = min(
            frame_width,
            round(max_template_width + 2 * (self.roi_margin_px + coarse_uncertainty)),
        )
        roi_height = min(
            frame_height,
            round(max_template_height + 2 * (self.roi_margin_px + coarse_uncertainty)),
        )

        rois: list[Roi] = []
        for index, candidate in enumerate(candidates, start=1):
            center_x = round((candidate.x + candidate.template.width / 2) / self.frame_downscale)
            center_y = round((candidate.y + candidate.template.height / 2) / self.frame_downscale)
            x = max(0, min(frame_width - roi_width, center_x - roi_width // 2))
            y = max(0, min(frame_height - roi_height, center_y - roi_height // 2))
            rois.append(Roi(index, x, y, roi_width, roi_height, candidate))
        return rois

    def _find_detail_matches(
        self,
        original_features: torch.Tensor,
        rois: list[Roi],
        *,
        collect_variance_filters: bool,
    ) -> tuple[list[DetailMatch], list[DetailVarianceFilter]]:
        variance_filters: list[DetailVarianceFilter] = []
        pending_matches: list[tuple[Roi, Template, int, ScoreMap, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for roi in rois:
            roi_features = original_features[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            for template in self.detail_templates:
                if template.height > roi.height or template.width > roi.width:
                    continue
                stride = self.detail_stride_for_scale(template.scale)
                score_map = _score_map(
                    roi_features,
                    template,
                    stride,
                    self.channel_weights,
                    variance_channel_weights=self.variance_channel_weights,
                    variance_log_distance_max=self.variance_log_distance_max,
                    variance_epsilon=self.variance_epsilon,
                    min_chroma_variance=self.min_chroma_variance,
                    min_brightness_variance=self.min_brightness_variance,
                )
                if (
                    score_map.variance_passes is None
                    or score_map.flatness_passes is None
                    or score_map.relative_variance_passes is None
                    or score_map.candidate_variances is None
                ):
                    raise RuntimeError("detail score map must include variance filter results")
                if collect_variance_filters:
                    variance_filters.append(
                        DetailVarianceFilter(
                            roi=roi,
                            template=template,
                            x_positions=score_map.x_positions,
                            y_positions=score_map.y_positions,
                            variance_passes=score_map.variance_passes.detach().cpu().numpy(),
                            flatness_passes=score_map.flatness_passes.detach().cpu().numpy(),
                            relative_variance_passes=(
                                score_map.relative_variance_passes.detach().cpu().numpy()
                            ),
                            candidate_variances=(
                                score_map.candidate_variances.detach().cpu().numpy()
                            ),
                            min_chroma_variance=self.min_chroma_variance,
                            min_brightness_variance=self.min_brightness_variance,
                            stride=stride,
                        )
                    )
                flat_scores = score_map.scores.flatten()
                best_index = torch.argmin(flat_scores)
                pending_matches.append(
                    (
                        roi,
                        template,
                        stride,
                        score_map,
                        best_index,
                        flat_scores[best_index],
                        score_map.variance_distances.flatten()[best_index],
                    )
                )

        if not pending_matches:
            return [], variance_filters

        # 詳細候補のインデックス・スコア・分散距離をまとめてCPUへ渡す。一候補ごとの
        # item() や bool判定によるGPU同期を避け、全GPUカーネルを先に投入できる。
        compact_results = torch.stack(
            [
                torch.stack((item[4].to(torch.float32), item[5], item[6]))
                for item in pending_matches
            ]
        ).detach().cpu().numpy()
        matches: list[DetailMatch] = []
        for (roi, template, stride, score_map, _, _, _), (best_index, score, variance_distance) in zip(
            pending_matches, compact_results
        ):
            if not np.isfinite(score):
                continue
            index = int(best_index)
            local_x = score_map.x_positions[index % len(score_map.x_positions)]
            local_y = score_map.y_positions[index // len(score_map.x_positions)]
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


def _to_gpu_bgr(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image_bgr)).to(device=device, dtype=torch.float32)


def _normalise_weights(values: Sequence[float], device: torch.device) -> torch.Tensor:
    weights = torch.as_tensor(values, dtype=torch.float32, device=device)
    if weights.shape != (3,) or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("brightness_weights must contain three non-negative values")
    return weights / weights.sum()


def _normalise_channel_weights(values: Sequence[float], device: torch.device) -> torch.Tensor:
    weights = torch.as_tensor(values, dtype=torch.float32, device=device)
    if weights.shape != (3,) or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("channel_weights must contain three non-negative values")
    return weights


def _bgr_to_features(image_bgr: torch.Tensor, brightness_weights: torch.Tensor) -> torch.Tensor:
    blue, green, red = image_bgr[..., 0], image_bgr[..., 1], image_bgr[..., 2]
    rg = red - green
    by = (red + green) * 0.5 - blue
    brightness = brightness_weights[0] * red + brightness_weights[1] * green + brightness_weights[2] * blue
    return torch.stack((rg, by, brightness), dim=-1)


def _center_weight(height: int, width: int, min_weight: float, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    distance = torch.sqrt(y[:, None].square() + x[None, :].square()) / (2.0**0.5)
    return min_weight + (1.0 - min_weight) * (1.0 - torch.clamp(distance, 0.0, 1.0))


def _downscale_on_gpu(image_bgr: torch.Tensor, scale: float) -> torch.Tensor:
    height = max(1, round(image_bgr.shape[0] * scale))
    width = max(1, round(image_bgr.shape[1] * scale))
    chw = image_bgr.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(chw, size=(height, width), mode="area").squeeze(0).permute(1, 2, 0)


def _scan_positions(maximum: int, stride: int) -> list[int]:
    positions = list(range(0, maximum + 1, stride))
    if positions[-1] != maximum:
        positions.append(maximum)
    return positions


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


def _score_map(
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
    x_positions = _scan_positions(frame_features.shape[1] - template.width, stride)
    y_positions = _scan_positions(frame_features.shape[0] - template.height, stride)
    candidates = _extract_candidate_patches(
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
    # 色・明度のどちらかに十分な空間変化があれば、平坦領域とは扱わない。
    flatness_passes = (chroma_variance >= min_chroma_variance) | (
        candidate_variance[:, 2] >= min_brightness_variance
    )
    variance_passes = relative_variance_passes & flatness_passes
    difference = template.features.unsqueeze(0) - candidates
    pixel_error = torch.sqrt(torch.sum(difference.square() * channel_weights, dim=3))
    unfiltered_scores = torch.sum(pixel_error * template.weights, dim=(1, 2)) / template.weights.sum()
    scores = torch.where(
        variance_passes,
        unfiltered_scores,
        torch.full_like(unfiltered_scores, float("inf")),
    )
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


def _extract_candidate_patches(
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
    patches = frame_features[
        y_indices[:, None, :, None],
        x_indices[None, :, None, :],
    ]
    return patches.reshape(len(y_positions) * len(x_positions), height, width, 3)
