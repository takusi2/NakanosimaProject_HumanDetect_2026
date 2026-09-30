"""粗探索・詳細探索の間で受け渡す結果データの型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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
        return int(np.count_nonzero(~self.flatness_passes))

    @property
    def relative_variance_rejected_count(self) -> int:
        return int(np.count_nonzero(self.flatness_passes & ~self.relative_variance_passes))

    @property
    def candidate_variance_mean(self) -> np.ndarray:
        return self.candidate_variances.mean(axis=(0, 1))


@dataclass(frozen=True)
class FrameResult:
    """1フレームの粗探索・詳細照合の全結果。"""

    coarse_image_bgr: np.ndarray | None
    coarse_candidates: list[Candidate]
    rois: list[Roi]
    detail_matches: list[DetailMatch]
    detail_variance_filters: list[DetailVarianceFilter]
    best_match: DetailMatch | None
    detected: bool
    temporal_status: str | None = None
    track_id: int | None = None
    positive_frames: int | None = None
    history_window_frames: int | None = None
