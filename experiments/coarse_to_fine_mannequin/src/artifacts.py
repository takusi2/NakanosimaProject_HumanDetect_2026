"""実験の中間画像とスコアを実行IDごとに保存する。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .pipeline import FrameResult, Template


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

        detail_records = []
        for detail in result.detail_matches:
            roi_image = original_bgr[
                detail.roi.y : detail.roi.y + detail.roi.height,
                detail.roi.x : detail.roi.x + detail.roi.width,
            ].copy()
            _draw_box(
                roi_image,
                detail.x - detail.roi.x,
                detail.y - detail.roi.y,
                detail.template.width,
                detail.template.height,
                (0, 255, 0),
                f"score={detail.score:.2f} scale={detail.template.scale:.3f}",
            )
            suffix = f"roi_{detail.roi.index}_scale_{detail.template.scale:.3f}"
            _write_image(self.detail_dir / f"{frame_id}_{suffix}.png", roi_image)
            selected = detail is result.best_match
            detail_records.append(
                {
                    "roi": detail.roi.index,
                    "scale": detail.template.scale,
                    "x": detail.x,
                    "y": detail.y,
                    "score": detail.score,
                    "variance_distance": detail.variance_distance,
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


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    encoded.tofile(str(path))
