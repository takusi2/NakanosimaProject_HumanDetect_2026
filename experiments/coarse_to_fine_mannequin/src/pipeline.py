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


@dataclass(frozen=True)
class FrameResult:
    """1フレームの粗探索・詳細照合の全結果。"""

    coarse_image_bgr: np.ndarray
    coarse_candidates: list[Candidate]
    rois: list[Roi]
    detail_matches: list[DetailMatch]
    best_match: DetailMatch
    detected: bool


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
        detail_stride: int = 4,
        coarse_top_k: int = 3,
        coarse_candidates_per_template: int = 20,
        nms_distance_original_px: int = 120,
        roi_margin_px: int = 32,
        final_max_error: float = 30.0,
        brightness_weights: Sequence[float] = (0.299, 0.587, 0.114),
        channel_weights: Sequence[float] = (0.5, 0.5, 1.0),
        min_weight: float = 0.05,
    ) -> None:
        if not 0.0 < frame_downscale < 1.0:
            raise ValueError("frame_downscale must be in the range (0, 1)")
        if coarse_stride <= 0 or detail_stride <= 0:
            raise ValueError("strides must be positive")
        if coarse_top_k <= 0 or coarse_candidates_per_template <= 0:
            raise ValueError("candidate counts must be positive")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA GPU and CUDA PyTorch installation.")
        self.frame_downscale = frame_downscale
        self.coarse_stride = coarse_stride
        self.detail_stride = detail_stride
        self.coarse_top_k = coarse_top_k
        self.coarse_candidates_per_template = coarse_candidates_per_template
        self.nms_distance_original_px = nms_distance_original_px
        self.roi_margin_px = roi_margin_px
        self.final_max_error = final_max_error
        self.min_weight = min_weight
        self.brightness_weights = _normalise_weights(brightness_weights, self.device)
        self.channel_weights = _normalise_channel_weights(channel_weights, self.device)

        self.coarse_templates = self._create_templates(template_bgr, coarse_template_scales, "coarse")
        self.detail_templates = self._create_templates(template_bgr, detail_template_scales, "detail")
        if not self.coarse_templates or not self.detail_templates:
            raise ValueError("at least one coarse and one detail template are required")

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
            templates.append(Template(f"{prefix}_{scale:.3f}", scale, image, features, weights))
        return templates

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        """1フレームをGPUで粗探索・詳細照合し、全中間結果を返す。"""
        with torch.inference_mode():
            # 元フレームは1回だけGPUへ転送する。詳細照合はこの特徴量のROIを切り出す。
            original_bgr_gpu = _to_gpu_bgr(frame_bgr, self.device)
            original_features = _bgr_to_features(original_bgr_gpu, self.brightness_weights)

            coarse_bgr_gpu = _downscale_on_gpu(original_bgr_gpu, self.frame_downscale)
            coarse_features = _bgr_to_features(coarse_bgr_gpu, self.brightness_weights)
            coarse_image_bgr = (
                coarse_bgr_gpu.round().to(torch.uint8).detach().cpu().numpy()
            )

            coarse_candidates = self._find_coarse_candidates(coarse_features)
            rois = self._make_rois(coarse_candidates, frame_bgr.shape[:2])
            detail_matches = self._find_detail_matches(original_features, rois)
            if not detail_matches:
                raise ValueError("no detail template fits inside the candidate ROIs")
            best_match = min(detail_matches, key=lambda item: item.score)
            return FrameResult(
                coarse_image_bgr=coarse_image_bgr,
                coarse_candidates=coarse_candidates,
                rois=rois,
                detail_matches=detail_matches,
                best_match=best_match,
                detected=best_match.score <= self.final_max_error,
            )

    def _find_coarse_candidates(self, coarse_features: torch.Tensor) -> list[Candidate]:
        all_candidates: list[Candidate] = []
        for template in self.coarse_templates:
            if template.height > coarse_features.shape[0] or template.width > coarse_features.shape[1]:
                continue
            scores, x_positions, y_positions = _score_map(
                coarse_features, template, self.coarse_stride, self.channel_weights
            )
            count = min(self.coarse_candidates_per_template, scores.numel())
            values, indices = torch.topk(scores.flatten(), k=count, largest=False)
            for value, index in zip(values, indices):
                flat_index = int(index.item())
                x = x_positions[flat_index % len(x_positions)]
                y = y_positions[flat_index // len(x_positions)]
                all_candidates.append(Candidate(template, x, y, float(value.item())))

        all_candidates.sort(key=lambda item: item.score)
        selected: list[Candidate] = []
        for candidate in all_candidates:
            center_x = (candidate.x + candidate.template.width / 2) / self.frame_downscale
            center_y = (candidate.y + candidate.template.height / 2) / self.frame_downscale
            is_far_enough = all(
                (center_x - (saved.x + saved.template.width / 2) / self.frame_downscale) ** 2
                + (center_y - (saved.y + saved.template.height / 2) / self.frame_downscale) ** 2
                >= self.nms_distance_original_px**2
                for saved in selected
            )
            if is_far_enough:
                selected.append(candidate)
            if len(selected) == self.coarse_top_k:
                break
        if not selected:
            raise ValueError("no coarse template fits inside the downscaled frame")
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
        self, original_features: torch.Tensor, rois: list[Roi]
    ) -> list[DetailMatch]:
        matches: list[DetailMatch] = []
        for roi in rois:
            roi_features = original_features[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            for template in self.detail_templates:
                if template.height > roi.height or template.width > roi.width:
                    continue
                scores, x_positions, y_positions = _score_map(
                    roi_features, template, self.detail_stride, self.channel_weights
                )
                best_index = int(torch.argmin(scores).item())
                local_x = x_positions[best_index % len(x_positions)]
                local_y = y_positions[best_index // len(x_positions)]
                matches.append(
                    DetailMatch(
                        roi,
                        template,
                        roi.x + local_x,
                        roi.y + local_y,
                        float(scores.flatten()[best_index].item()),
                    )
                )
        return matches


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


def _score_map(
    frame_features: torch.Tensor,
    template: Template,
    stride: int,
    channel_weights: torch.Tensor,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """1テンプレートについて全配置候補の重み付き色差をGPUで求める。"""
    x_positions = _scan_positions(frame_features.shape[1] - template.width, stride)
    y_positions = _scan_positions(frame_features.shape[0] - template.height, stride)
    candidates = torch.stack(
        [
            frame_features[y : y + template.height, x : x + template.width]
            for y in y_positions
            for x in x_positions
        ]
    )
    difference = template.features.unsqueeze(0) - candidates
    pixel_error = torch.sqrt(torch.sum(difference.square() * channel_weights, dim=3))
    scores = torch.sum(pixel_error * template.weights, dim=(1, 2)) / template.weights.sum()
    return scores.reshape(len(y_positions), len(x_positions)), x_positions, y_positions
