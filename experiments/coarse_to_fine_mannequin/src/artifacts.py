"""実験の中間画像とスコアを実行IDごとに保存する。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .pipeline import DetailVarianceFilter, FrameResult, Template


class ArtifactWriter:
    """ユーザーが検証したい全段階の成果物をresults配下へ書き出す。"""

    def __init__(self, results_root: Path) -> None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = results_root / run_id
        self.template_dir = self.run_dir / "01_detail_templates"
        self.frame_dir = self.run_dir / "02_frames"
        self.coarse_dir = self.run_dir / "03_coarse_match"
        self.roi_dir = self.run_dir / "04_coarse_rois_original"
        self.detail_dir = self.run_dir / "05_detail_match"
        self.score_dir = self.run_dir / "06_scores"
        for directory in (
            self.template_dir,
            self.frame_dir,
            self.coarse_dir,
            self.roi_dir,
            self.detail_dir,
            self.score_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.score_dir / "scores.csv").open("w", newline="", encoding="utf-8")
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

    def close(self) -> None:
        self.csv_file.close()

    def save_detail_templates(self, templates: list[Template]) -> None:
        for template in templates:
            _write_image(
                self.template_dir / f"template_scale_{template.scale:.3f}.png", template.image_bgr
            )

    def save_frame(self, frame_number: int, original_bgr: np.ndarray, result: FrameResult) -> None:
        frame_id = f"frame_{frame_number:06d}"
        _write_image(self.frame_dir / f"{frame_id}_original.png", original_bgr)
        _write_image(self.frame_dir / f"{frame_id}_coarse.png", result.coarse_image_bgr)

        coarse_records = []
        for rank, candidate in enumerate(result.coarse_candidates, start=1):
            image = result.coarse_image_bgr.copy()
            _draw_box(
                image,
                candidate.x,
                candidate.y,
                candidate.template.width,
                candidate.template.height,
                (0, 255, 255),
                f"rank={rank} score={candidate.score:.2f} scale={candidate.template.scale:.3f}",
            )
            _write_image(self.coarse_dir / f"{frame_id}_rank_{rank}.png", image)
            coarse_records.append(
                {
                    "rank": rank,
                    "scale": candidate.template.scale,
                    "x": candidate.x,
                    "y": candidate.y,
                    "score": candidate.score,
                }
            )
            self.csv_writer.writerow(
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
        _write_image(self.roi_dir / f"{frame_id}_rois.png", roi_image)

        variance_filter_records = []
        for variance_filter in result.detail_variance_filters:
            roi_image = original_bgr[
                variance_filter.roi.y : variance_filter.roi.y + variance_filter.roi.height,
                variance_filter.roi.x : variance_filter.roi.x + variance_filter.roi.width,
            ].copy()
            filter_image = _draw_variance_rejections(roi_image, variance_filter)
            suffix = (
                f"roi_{variance_filter.roi.index}_scale_{variance_filter.template.scale:.3f}"
            )
            _write_image(self.detail_dir / f"{frame_id}_{suffix}_variance_filter.png", filter_image)
            variance_filter_records.append(
                {
                    "roi": variance_filter.roi.index,
                    "scale": variance_filter.template.scale,
                    "candidates_total": variance_filter.candidate_count,
                    "candidates_passed": variance_filter.passed_count,
                    "candidates_rejected": variance_filter.rejected_count,
                    "stride": variance_filter.stride,
                }
            )

        detail_records = []
        for detail in result.detail_matches:
            roi_image = original_bgr[
                detail.roi.y : detail.roi.y + detail.roi.height,
                detail.roi.x : detail.roi.x + detail.roi.width,
            ].copy()
            is_best = detail is result.best_match
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
            _write_image(self.detail_dir / f"{frame_id}_{suffix}.png", roi_image)
            selected = is_best
            detail_records.append(
                {
                    "roi": detail.roi.index,
                    "scale": detail.template.scale,
                    "x": detail.x,
                    "y": detail.y,
                    "score": detail.score,
                    "variance_distance": detail.variance_distance,
                    "stride": detail.stride,
                    "selected": selected,
                }
            )
            self.csv_writer.writerow(
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
        _write_image(self.detail_dir / f"{frame_id}_best.png", best_image)
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
        (self.score_dir / f"{frame_id}.json").write_text(
            json.dumps(score_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _draw_box(
    image: np.ndarray, x: int, y: int, width: int, height: int, colour: tuple[int, int, int], label: str
) -> None:
    cv2.rectangle(image, (x, y), (x + width, y + height), colour, 2)
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 500), 26), (0, 0, 0), -1)
    cv2.putText(image, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _draw_variance_rejections(image: np.ndarray, variance_filter: DetailVarianceFilter) -> np.ndarray:
    """分散フィルタで除外された候補領域全体をオレンジ枠で示す。"""
    rejected_y_indices, rejected_x_indices = np.where(~variance_filter.variance_passes)
    result = image.copy()
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
            (0, 165, 255),
            1,
        )
    label = (
        "VARIANCE REJECTED (orange candidate boxes): "
        f"{variance_filter.rejected_count}/{variance_filter.candidate_count}"
    )
    _draw_box(result, 0, 0, 0, 0, (0, 165, 255), label)
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
