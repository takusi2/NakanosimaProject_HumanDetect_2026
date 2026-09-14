from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from experiments.coarse_to_fine_mannequin.src.pipeline import CoarseToFineMatcher
from experiments.coarse_to_fine_mannequin.src.artifacts import (
    AnnotatedVideoWriter,
    ArtifactSaveOptions,
    ArtifactWriter,
)
from experiments.coarse_to_fine_mannequin.src.performance import (
    FrameTiming,
    PerformanceOptions,
    PerformanceTracker,
)


class CoarseToFineMatcherTests(unittest.TestCase):
    def test_finds_template_after_coarse_roi_selection(self) -> None:
        rng = np.random.default_rng(123)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        frame = np.zeros((80, 96, 3), dtype=np.uint8)
        frame[40:56, 48:60] = template
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
            final_max_error=0.1,
        )

        result = matcher.process(frame)

        self.assertIsNotNone(result.best_match)
        self.assertEqual((result.best_match.x, result.best_match.y), (48, 40))
        self.assertTrue(result.detected)
        self.assertEqual(result.coarse_image_bgr.shape[:2], (20, 24))
        self.assertEqual(len(result.rois), 1)

    def test_rejects_uniform_wall_by_variance(self) -> None:
        rng = np.random.default_rng(456)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        wall_frame = np.full((80, 96, 3), 230, dtype=np.uint8)
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
            variance_log_distance_max=3.0,
        )

        result = matcher.process(wall_frame)

        self.assertFalse(result.detected)
        self.assertIsNone(result.best_match)
        self.assertEqual(len(result.detail_variance_filters), 1)
        variance_filter = result.detail_variance_filters[0]
        self.assertEqual(variance_filter.passed_count, 0)
        self.assertEqual(variance_filter.rejected_count, variance_filter.candidate_count)

    def test_rejects_uniform_wall_by_flatness_filter(self) -> None:
        rng = np.random.default_rng(654)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        wall_frame = np.full((80, 96, 3), 230, dtype=np.uint8)
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
            # 相対分散差ではなく、絶対分散下限による除外だけを検証する。
            variance_log_distance_max=100.0,
            min_chroma_variance=1.0,
            min_brightness_variance=1.0,
        )

        result = matcher.process(wall_frame)

        self.assertIsNone(result.best_match)
        variance_filter = result.detail_variance_filters[0]
        self.assertEqual(
            variance_filter.flatness_rejected_count, variance_filter.candidate_count
        )
        self.assertEqual(variance_filter.relative_variance_rejected_count, 0)

    def test_skips_cpu_artifact_data_without_changing_detection(self) -> None:
        rng = np.random.default_rng(357)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        frame = np.zeros((80, 96, 3), dtype=np.uint8)
        frame[40:56, 48:60] = template
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
            final_max_error=0.1,
        )

        full_result = matcher.process(frame)
        lightweight_result = matcher.process(
            frame,
            collect_coarse_image=False,
            collect_detail_variance_filters=False,
        )

        self.assertTrue(lightweight_result.detected)
        self.assertIsNone(lightweight_result.coarse_image_bgr)
        self.assertEqual(lightweight_result.detail_variance_filters, [])
        self.assertIsNotNone(full_result.best_match)
        self.assertIsNotNone(lightweight_result.best_match)
        self.assertEqual(
            (lightweight_result.best_match.x, lightweight_result.best_match.y),
            (full_result.best_match.x, full_result.best_match.y),
        )
        self.assertEqual(lightweight_result.best_match.score, full_result.best_match.score)

    def test_saves_variance_filter_visualisation(self) -> None:
        rng = np.random.default_rng(789)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        wall_frame = np.full((80, 96, 3), 230, dtype=np.uint8)
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
            variance_log_distance_max=3.0,
        )
        result = matcher.process(wall_frame)

        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = ArtifactWriter(Path(temporary_directory))
            writer.save_frame(1, wall_frame, result)
            writer.close()

            run_directory = next(Path(temporary_directory).glob("run_*"))
            filter_images = list(
                (run_directory / "05_detail_match").glob("*_variance_filter.png")
            )
            self.assertEqual(len(filter_images), 1)
            self.assertGreater(filter_images[0].stat().st_size, 0)
            score_data = json.loads(
                (run_directory / "06_scores" / "frame_000001.json").read_text(encoding="utf-8")
            )
            variance_data = score_data["variance_filters"][0]
            self.assertEqual(variance_data["candidates_passed"], 0)
            self.assertGreater(variance_data["candidates_rejected"], 0)
            self.assertIn("flatness_rejected", variance_data)
            self.assertIn("relative_variance_rejected", variance_data)
            self.assertIn("candidate_variance_mean", variance_data)

    def test_disables_all_artifacts_without_creating_result_directory(self) -> None:
        rng = np.random.default_rng(246)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        frame = np.zeros((80, 96, 3), dtype=np.uint8)
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
        )
        result = matcher.process(frame)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            writer = ArtifactWriter(root, ArtifactSaveOptions(enabled=False))
            writer.save_detail_templates(matcher.detail_templates)
            writer.save_frame(1, frame, result)
            writer.close()

            self.assertIsNone(writer.run_dir)
            self.assertEqual(list(root.iterdir()), [])

    def test_saves_only_enabled_artifact_types(self) -> None:
        rng = np.random.default_rng(135)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        frame = np.zeros((80, 96, 3), dtype=np.uint8)
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
        )
        result = matcher.process(frame)
        options = ArtifactSaveOptions(
            detail_templates=False,
            original_frame=False,
            coarse_frame=False,
            coarse_candidates=False,
            coarse_rois=True,
            variance_filters=False,
            detail_matches=False,
            best_match=True,
            scores_csv=False,
            scores_json=True,
            annotated_video=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = ArtifactWriter(Path(temporary_directory), options)
            writer.save_detail_templates(matcher.detail_templates)
            writer.save_frame(1, frame, result)
            writer.close()

            run_directory = writer.require_run_dir()
            self.assertTrue((run_directory / "04_coarse_rois_original" / "frame_000001_rois.png").exists())
            self.assertTrue((run_directory / "05_detail_match" / "frame_000001_best.png").exists())
            self.assertTrue((run_directory / "06_scores" / "frame_000001.json").exists())
            self.assertFalse((run_directory / "01_detail_templates").exists())
            self.assertFalse((run_directory / "02_frames").exists())
            self.assertFalse((run_directory / "03_coarse_match").exists())
            self.assertFalse((run_directory / "06_scores" / "scores.csv").exists())

    def test_reads_legacy_and_new_save_configurations(self) -> None:
        legacy = ArtifactSaveOptions.from_config(
            {"save_every_n_frames": 3, "save_annotated_video": False}
        )
        self.assertEqual(legacy.every_n_frames, 3)
        self.assertFalse(legacy.annotated_video)

        configured = ArtifactSaveOptions.from_config(
            {
                "save": {
                    "enabled": True,
                    "every_n_frames": 4,
                    "images": {"coarse_rois": False},
                    "scores": {"csv": False, "json": True},
                    "annotated_video": False,
                }
            }
        )
        self.assertEqual(configured.every_n_frames, 4)
        self.assertFalse(configured.coarse_rois)
        self.assertFalse(configured.scores_csv)
        self.assertTrue(configured.scores_json)
        self.assertFalse(configured.annotated_video)

    def test_performance_tracker_excludes_warmup_and_reports_fps(self) -> None:
        tracker = PerformanceTracker(
            PerformanceOptions(enabled=True, warmup_frames=1, report_every_n_frames=2)
        )
        for frame_number, total_ms in ((1, 100.0), (2, 20.0), (3, 40.0)):
            tracker.record(
                FrameTiming(
                    frame_number=frame_number,
                    decode_ms=2.0,
                    detection_ms=10.0,
                    artifact_write_ms=3.0,
                    display_ms=5.0,
                    total_ms=total_ms,
                )
            )

        summary = tracker.summary()
        self.assertEqual(summary["frames_measured"], 2)
        self.assertEqual(summary["total"]["mean_ms"], 30.0)
        self.assertAlmostEqual(summary["total"]["mean_fps"], 1000.0 / 30.0)
        self.assertTrue(tracker.should_report(2))
        self.assertIn("artifact_write", tracker.format_summary())

    def test_saves_annotated_video_with_rois_and_best_match(self) -> None:
        rng = np.random.default_rng(987)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        frame = np.zeros((80, 96, 3), dtype=np.uint8)
        frame[40:56, 48:60] = template
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride_base=1,
            detail_stride_min=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
            final_max_error=0.1,
        )
        result = matcher.process(frame)

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_writer = AnnotatedVideoWriter(
                Path(temporary_directory), fps=30.0, frame_width=96, frame_height=80
            )
            video_writer.write(1, frame, result)
            video_path = video_writer.path
            video_writer.close()

            self.assertGreater(video_path.stat().st_size, 0)
            capture = cv2.VideoCapture(str(video_path))
            try:
                self.assertTrue(capture.isOpened())
                ok, saved_frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(saved_frame.shape[:2], (80, 96))
            finally:
                capture.release()

    def test_uses_smaller_stride_for_smaller_detail_template(self) -> None:
        template = np.full((20, 20, 3), 128, dtype=np.uint8)
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0, 0.9, 0.8, 0.7, 0.6],
            detail_stride_base=4,
            detail_stride_min=2,
        )

        strides = [matcher.detail_stride_for_scale(item.scale) for item in matcher.detail_templates]

        self.assertEqual(strides, [4, 3, 3, 2, 2])


if __name__ == "__main__":
    unittest.main()
