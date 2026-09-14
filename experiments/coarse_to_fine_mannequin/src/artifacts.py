"""実験の中間画像とスコアを実行IDごとに保存する。"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .pipeline import DetailVarianceFilter, FrameResult, Template


@dataclass(frozen=True)
class ArtifactSaveOptions:
    """実験成果物を種類ごとに選んで保存するための設定。"""

    enabled: bool = True
    every_n_frames: int = 1
    detail_templates: bool = True
    original_frame: bool = True
    coarse_frame: bool = True
    coarse_candidates: bool = True
    coarse_rois: bool = True
    variance_filters: bool = True
    detail_matches: bool = True
    best_match: bool = True
    scores_csv: bool = True
    scores_json: bool = True
    annotated_video: bool = True
    performance_json: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "ArtifactSaveOptions":
        """新しい ``save:`` 設定と旧設定の両方を読み取る。"""
        raw_save = config.get("save", {})
        if raw_save is None:
            raw_save = {}
        if not isinstance(raw_save, Mapping):
            raise ValueError("save must be a YAML mapping")

        def nested(name: str) -> Mapping[str, object]:
            value = raw_save.get(name, {})
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ValueError(f"save.{name} must be a YAML mapping")
            return value

        images = nested("images")
        scores = nested("scores")
        every_n_frames = int(raw_save.get("every_n_frames", config.get("save_every_n_frames", 1)))
        if every_n_frames <= 0:
            raise ValueError("save.every_n_frames must be positive")
        return cls(
            enabled=bool(raw_save.get("enabled", True)),
            every_n_frames=every_n_frames,
            detail_templates=bool(raw_save.get("detail_templates", True)),
            original_frame=bool(images.get("original_frame", True)),
            coarse_frame=bool(images.get("coarse_frame", True)),
            coarse_candidates=bool(images.get("coarse_candidates", True)),
            coarse_rois=bool(images.get("coarse_rois", True)),
            variance_filters=bool(images.get("variance_filters", True)),
            detail_matches=bool(images.get("detail_matches", True)),
            best_match=bool(images.get("best_match", True)),
            scores_csv=bool(scores.get("csv", True)),
            scores_json=bool(scores.get("json", True)),
            annotated_video=bool(
                raw_save.get("annotated_video", config.get("save_annotated_video", True))
            ),
            performance_json=bool(raw_save.get("performance_json", True)),
        )

    @property
    def has_frame_output(self) -> bool:
        return self.enabled and any(
            (
                self.original_frame,
                self.coarse_frame,
                self.coarse_candidates,
                self.coarse_rois,
                self.variance_filters,
                self.detail_matches,
                self.best_match,
                self.scores_csv,
                self.scores_json,
            )
        )

    @property
    def has_any_output(self) -> bool:
        return self.enabled and (
            self.detail_templates
            or self.has_frame_output
            or self.annotated_video
            or self.performance_json
        )

    def should_save_frame(self, frame_number: int) -> bool:
        return self.has_frame_output and frame_number % self.every_n_frames == 0


class ArtifactWriter:
    """設定で選択された実験成果物だけをresults配下へ書き出す。"""

    def __init__(self, results_root: Path, options: ArtifactSaveOptions | None = None) -> None:
        self.options = options or ArtifactSaveOptions()
        self.run_dir: Path | None = None
        self.csv_file = None
        self.csv_writer: csv.DictWriter | None = None
        if self.options.has_any_output:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self.run_dir = results_root / run_id
            self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.options.enabled and self.options.scores_csv:
            score_dir = self._directory("06_scores")
            self.csv_file = (score_dir / "scores.csv").open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=(
                    "frame",
                    "stage",
                    "roi",
                    "template_scale",
                    "x",
                    "y",
                    "score",
                    "variance_distance",
                    "stride",
                    "selected",
                ),
            )
            self.csv_writer.writeheader()

    def _directory(self, name: str) -> Path:
        if self.run_dir is None:
            raise RuntimeError("output is disabled; no result directory was created")
        directory = self.run_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def require_run_dir(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("output is disabled; no result directory was created")
        return self.run_dir

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()

    def save_detail_templates(self, templates: list[Template]) -> None:
        if not (self.options.enabled and self.options.detail_templates):
            return
        template_dir = self._directory("01_detail_templates")
        for template in templates:
            _write_image(
                template_dir / f"template_scale_{template.scale:.3f}.png", template.image_bgr
            )

    def save_frame(self, frame_number: int, original_bgr: np.ndarray, result: FrameResult) -> None:
        if not self.options.should_save_frame(frame_number):
            return
        frame_id = f"frame_{frame_number:06d}"
        coarse_image = result.coarse_image_bgr
        if (self.options.coarse_frame or self.options.coarse_candidates) and coarse_image is None:
            raise RuntimeError("coarse image was not collected for a selected artifact")
        if self.options.original_frame:
            _write_image(self._directory("02_frames") / f"{frame_id}_original.png", original_bgr)
        if self.options.coarse_frame:
            _write_image(self._directory("02_frames") / f"{frame_id}_coarse.png", coarse_image)

        coarse_records = []
        for rank, candidate in enumerate(result.coarse_candidates, start=1):
            record = {
                "rank": rank,
                "scale": candidate.template.scale,
                "x": candidate.x,
                "y": candidate.y,
                "score": candidate.score,
            }
            if self.options.scores_json:
                coarse_records.append(record)
            if self.options.coarse_candidates:
                image = coarse_image.copy()
                _draw_box(
                    image,
                    candidate.x,
                    candidate.y,
                    candidate.template.width,
                    candidate.template.height,
                    (0, 255, 255),
                    f"rank={rank} score={candidate.score:.2f} scale={candidate.template.scale:.3f}",
                )
                _write_image(self._directory("03_coarse_match") / f"{frame_id}_rank_{rank}.png", image)
            self._write_csv(
                {
                    "frame": frame_number,
                    "stage": "coarse",
                    "roi": rank,
                    "template_scale": candidate.template.scale,
                    "x": candidate.x,
                    "y": candidate.y,
                    "score": candidate.score,
                    "variance_distance": "",
                    "stride": "",
                    "selected": True,
                }
            )

        if self.options.coarse_rois:
            roi_image = original_bgr.copy()
            for roi in result.rois:
                _draw_box(
                    roi_image,
                    roi.x,
                    roi.y,
                    roi.width,
                    roi.height,
                    (255, 255, 0),
                    f"ROI {roi.index} coarse={roi.coarse_candidate.score:.2f}",
                )
            _write_image(self._directory("04_coarse_rois_original") / f"{frame_id}_rois.png", roi_image)

        variance_filter_records = []
        for variance_filter in result.detail_variance_filters:
            if self.options.variance_filters:
                roi_image = original_bgr[
                    variance_filter.roi.y : variance_filter.roi.y + variance_filter.roi.height,
                    variance_filter.roi.x : variance_filter.roi.x + variance_filter.roi.width,
                ].copy()
                filter_image = _draw_variance_rejections(roi_image, variance_filter)
                suffix = (
                    f"roi_{variance_filter.roi.index}_scale_{variance_filter.template.scale:.3f}"
                )
                _write_image(
                    self._directory("05_detail_match")
                    / f"{frame_id}_{suffix}_variance_filter.png",
                    filter_image,
                )
            if self.options.scores_json:
                variance_filter_records.append(
                    {
                        "roi": variance_filter.roi.index,
                        "scale": variance_filter.template.scale,
                        "candidates_total": variance_filter.candidate_count,
                        "candidates_passed": variance_filter.passed_count,
                        "candidates_rejected": variance_filter.rejected_count,
                        "flatness_rejected": variance_filter.flatness_rejected_count,
                        "relative_variance_rejected": (
                            variance_filter.relative_variance_rejected_count
                        ),
                        "candidate_variance_mean": {
                            "rg": float(variance_filter.candidate_variance_mean[0]),
                            "by": float(variance_filter.candidate_variance_mean[1]),
                            "brightness": float(variance_filter.candidate_variance_mean[2]),
                        },
                        "min_chroma_variance": variance_filter.min_chroma_variance,
                        "min_brightness_variance": variance_filter.min_brightness_variance,
                        "stride": variance_filter.stride,
                    }
                )

        detail_records = []
        for detail in result.detail_matches:
            is_best = detail is result.best_match
            selected = is_best
            record = {
                "roi": detail.roi.index,
                "scale": detail.template.scale,
                "x": detail.x,
                "y": detail.y,
                "score": detail.score,
                "variance_distance": detail.variance_distance,
                "stride": detail.stride,
                "selected": selected,
            }
            if self.options.scores_json:
                detail_records.append(record)
            if self.options.detail_matches:
                roi_image = original_bgr[
                    detail.roi.y : detail.roi.y + detail.roi.height,
                    detail.roi.x : detail.roi.x + detail.roi.width,
                ].copy()
                if is_best and result.detected:
                    colour = (0, 255, 0)
                    label = f"MATCH score={detail.score:.2f} scale={detail.template.scale:.3f}"
                elif is_best:
                    colour = (0, 0, 255)
                    label = (
                        f"CLOSEST REJECTED score={detail.score:.2f} "
                        f"scale={detail.template.scale:.3f}"
                    )
                else:
                    colour = (0, 255, 255)
                    label = f"DETAIL CANDIDATE score={detail.score:.2f} scale={detail.template.scale:.3f}"
                _draw_box(
                    roi_image,
                    detail.x - detail.roi.x,
                    detail.y - detail.roi.y,
                    detail.template.width,
                    detail.template.height,
                    colour,
                    label,
                )
                suffix = f"roi_{detail.roi.index}_scale_{detail.template.scale:.3f}"
                _write_image(self._directory("05_detail_match") / f"{frame_id}_{suffix}.png", roi_image)
            self._write_csv(
                {
                    "frame": frame_number,
                    "stage": "detail",
                    "roi": detail.roi.index,
                    "template_scale": detail.template.scale,
                    "x": detail.x,
                    "y": detail.y,
                    "score": detail.score,
                    "variance_distance": detail.variance_distance,
                    "stride": detail.stride,
                    "selected": selected,
                }
            )

        if self.options.best_match:
            best_image = original_bgr.copy()
            best = result.best_match
            if best is None:
                _draw_box(
                    best_image,
                    0,
                    0,
                    0,
                    0,
                    (0, 0, 255),
                    "NO DETAIL MATCH: variance filter rejected all candidates",
                )
            else:
                _draw_box(
                    best_image,
                    best.x,
                    best.y,
                    best.template.width,
                    best.template.height,
                    (0, 255, 0) if result.detected else (0, 0, 255),
                    f"{'MANNEQUIN' if result.detected else 'NO MATCH'} score={best.score:.2f}",
                )
            _write_image(self._directory("05_detail_match") / f"{frame_id}_best.png", best_image)
        if self.csv_file is not None:
            self.csv_file.flush()

        score_json = {
            "frame": frame_number,
            "detected": result.detected,
            "coarse_candidates": coarse_records,
            "rois": [
                {"index": roi.index, "x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}
                for roi in result.rois
            ],
            "detail_matches": detail_records,
            "variance_filters": variance_filter_records,
            "best_match": (
                detail_records[[detail is result.best_match for detail in result.detail_matches].index(True)]
                if result.best_match is not None
                else None
            ),
        }
        if self.options.scores_json:
            (self._directory("06_scores") / f"{frame_id}.json").write_text(
                json.dumps(score_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _write_csv(self, row: dict[str, object]) -> None:
        if self.csv_writer is not None:
            self.csv_writer.writerow(row)

    def save_performance_summary(self, summary: dict[str, object]) -> Path | None:
        """計測が完了した後にのみ書き出すため、フレーム処理時間には含めない。"""
        if not (self.options.enabled and self.options.performance_json):
            return None
        path = self._directory("08_performance") / "performance_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _draw_box(
    image: np.ndarray, x: int, y: int, width: int, height: int, colour: tuple[int, int, int], label: str
) -> None:
    cv2.rectangle(image, (x, y), (x + width, y + height), colour, 2)
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 500), 26), (0, 0, 0), -1)
    cv2.putText(image, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _draw_variance_rejections(image: np.ndarray, variance_filter: DetailVarianceFilter) -> np.ndarray:
    """平坦領域・相対分散差で除外された候補領域全体を色分けして示す。"""
    result = image.copy()
    flatness_rejected = ~variance_filter.flatness_passes
    relative_rejected = (
        variance_filter.flatness_passes & ~variance_filter.relative_variance_passes
    )
    for rejection_mask, colour in (
        (relative_rejected, (0, 165, 255)),  # orange
        (flatness_rejected, (255, 0, 255)),  # magenta
    ):
        rejected_y_indices, rejected_x_indices = np.where(rejection_mask)
        for y_index, x_index in zip(rejected_y_indices, rejected_x_indices):
            x = variance_filter.x_positions[x_index]
            y = variance_filter.y_positions[y_index]
            cv2.rectangle(
                result,
                (x, y),
                (
                    min(image.shape[1] - 1, x + variance_filter.template.width - 1),
                    min(image.shape[0] - 1, y + variance_filter.template.height - 1),
                ),
                colour,
                1,
            )
    label = (
        "FLAT magenta="
        f"{variance_filter.flatness_rejected_count}, RELATIVE orange="
        f"{variance_filter.relative_variance_rejected_count}"
    )
    _draw_box(result, 0, 0, 0, 0, (255, 0, 255), label)
    return result


class AnnotatedVideoWriter:
    """粗探索ROIと最終候補を重ねた検証動画を書き出す。"""

    def __init__(self, run_dir: Path, fps: float, frame_width: int, frame_height: int) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("annotated video frame dimensions must be positive")
        self.video_dir = run_dir / "07_annotated_video"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.video_dir / "coarse_rois_and_best_match.mp4"
        fps = fps if fps > 0 else 30.0
        self.writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open annotated video for writing: {self.path}")

    def write(self, frame_number: int, original_bgr: np.ndarray, result: FrameResult) -> None:
        self.writer.write(draw_annotated_frame(frame_number, original_bgr, result))

    def close(self) -> None:
        self.writer.release()


def draw_annotated_frame(
    frame_number: int, original_bgr: np.ndarray, result: FrameResult
) -> np.ndarray:
    """元フレームに粗探索ROIと、最終的に最小誤差となった候補を重ねる。"""
    image = original_bgr.copy()
    for roi in result.rois:
        cv2.rectangle(
            image,
            (roi.x, roi.y),
            (roi.x + roi.width, roi.y + roi.height),
            (255, 0, 0),
            1,
        )
        cv2.putText(
            image,
            f"ROI {roi.index}",
            (roi.x + 2, min(image.shape[0] - 4, roi.y + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    best = result.best_match
    if best is None:
        status = f"F{frame_number} NO: variance rejected"
        colour = (0, 0, 255)
    else:
        colour = (0, 255, 0) if result.detected else (0, 0, 255)
        status = (
            f"F{frame_number} {'MATCH' if result.detected else 'NO'} "
            f"{best.score:.2f} x{best.template.scale:.2f}"
        )
        cv2.rectangle(
            image,
            (best.x, best.y),
            (best.x + best.template.width, best.y + best.template.height),
            colour,
            2,
        )

    font_scale = 0.32
    (text_width, text_height), baseline = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    banner_width = min(image.shape[1], text_width + 8)
    banner_height = min(image.shape[0], text_height + baseline + 6)
    cv2.rectangle(image, (0, 0), (banner_width, banner_height), (0, 0, 0), -1)
    cv2.putText(
        image,
        status,
        (4, text_height + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        1,
        cv2.LINE_AA,
    )
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    encoded.tofile(str(path))
