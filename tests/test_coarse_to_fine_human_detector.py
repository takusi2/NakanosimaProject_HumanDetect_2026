from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import yaml

import coarse_to_fine_human_detector as detector_module
from coarse_to_fine_human_detector import CoarseToFineHumanDetector


class _FakeDetector:
    def __init__(self, input_source: str) -> None:
        self.conf = SimpleNamespace(
            input_source=input_source,
            video_path="example.mp4",
            camera_index=7,
            frame_width=96,
            frame_height=100,
            window_name="test detector",
        )
        self.frames: list[tuple[np.ndarray, float | None]] = []
        self.display_sizes: list[tuple[int, int]] = []
        self.closed = False

    def process_frame(
        self, frame: np.ndarray, source_timestamp_s: float | None = None
    ) -> tuple[np.ndarray, None, None]:
        self.frames.append((frame, source_timestamp_s))
        return frame, None, None

    def close(self) -> None:
        self.closed = True

    def get_display_frame(self, display_width: int, display_height: int) -> np.ndarray:
        self.display_sizes.append((display_width, display_height))
        return cv2.resize(self.frames[-1][0], (display_width, display_height))


class _FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        return (self.read_count == 1, self.frame.copy() if self.read_count == 1 else None)

    def get(self, property_id: int) -> float:
        self.last_property_id = property_id
        return 1250.0

    def release(self) -> None:
        self.released = True


class _FakeInsightSensor:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.received = 0

    def receive_one_set(self) -> None:
        self.received += 1

    def get_center_surround_bgr_image(self) -> np.ndarray:
        return self.frame.copy()


class CoarseToFineHumanDetectorTests(unittest.TestCase):
    def _make_detector(
        self, directory: Path, *, final_max_error: float, confirm_frames: int = 1
    ) -> tuple[CoarseToFineHumanDetector, np.ndarray]:
        assets = directory / "assets"
        assets.mkdir()
        rng = np.random.default_rng(123)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        template_path = assets / "template.png"
        self.assertTrue(cv2.imwrite(str(template_path), template))
        config = {
            "input_source": "video",
            "video_path": "assets/input.mp4",
            "camera_index": 0,
            "frame_width": 96,
            "frame_height": 100,
            "window_name": "test detector",
            "template_path": "assets/template.png",
            "device": "cpu",
            "frame_downscale": 0.25,
            "coarse_template_scales": [0.25],
            "detail_template_scales": [1.0],
            "coarse_stride": 1,
            "detail_stride_base": 1,
            "detail_stride_min": 1,
            "coarse_top_k": 1,
            "coarse_candidates_per_template": 20,
            "nms_distance_original_px": 10,
            "roi_margin_px": 8,
            "final_max_error": final_max_error,
            "target_confirm_window_frames": 10,
            "target_confirm_frames": confirm_frames,
            "match_iou_thresh": 0.8,
            "variance_log_distance_max": 100.0,
            "min_chroma_variance": 0.0,
            "min_brightness_variance": 0.0,
        }
        config_path = directory / "detector.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return CoarseToFineHumanDetector(str(config_path)), template

    def test_match_returns_center_class_and_annotated_frame_without_artifact_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector, template = self._make_detector(Path(temporary_directory), final_max_error=0.1)
            frame = np.zeros((100, 96, 3), dtype=np.uint8)
            frame[60:76, 48:60] = template

            with mock.patch.object(detector.matcher, "process", wraps=detector.matcher.process) as process:
                output, center_x, class_name = detector.process_frame(frame)

            process.assert_called_once_with(
                frame,
                collect_coarse_image=False,
                collect_detail_variance_filters=False,
            )
            self.assertEqual(center_x, 54.0)
            self.assertEqual(class_name, "A")
            self.assertEqual(output.shape, frame.shape)
            self.assertFalse(np.array_equal(output, frame))
            self.assertEqual(Path(detector.conf.template_path), Path(temporary_directory) / "assets" / "template.png")

    def test_no_match_returns_none_and_draws_best_candidate_in_red(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector, template = self._make_detector(Path(temporary_directory), final_max_error=-0.1)
            frame = np.zeros((100, 96, 3), dtype=np.uint8)
            frame[60:76, 48:60] = template

            output, center_x, class_name = detector.process_frame(frame)

            self.assertIsNone(center_x)
            self.assertIsNone(class_name)
            self.assertTrue(np.any(np.all(output == (0, 0, 255), axis=2)))

    def test_confirms_same_position_after_five_matches_in_ten_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector, _ = self._make_detector(
                Path(temporary_directory), final_max_error=0.1, confirm_frames=5
            )
            template = SimpleNamespace(width=12, height=16, scale=1.0)
            raw_result = SimpleNamespace(
                detected=True,
                best_match=SimpleNamespace(x=48, y=60, score=0.0, template=template),
            )
            frame = np.zeros((100, 96, 3), dtype=np.uint8)

            with mock.patch.object(detector.matcher, "process", return_value=raw_result):
                for _ in range(4):
                    _, center_x, class_name = detector.process_frame(frame)
                    self.assertIsNone(center_x)
                    self.assertIsNone(class_name)
                _, center_x, class_name = detector.process_frame(frame)

            self.assertEqual(center_x, 54.0)
            self.assertEqual(class_name, "A")
            self.assertEqual(detector.track_info[0].target_history, [True] * 5)

    def test_distant_position_starts_a_new_maybe_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector, _ = self._make_detector(
                Path(temporary_directory), final_max_error=0.1, confirm_frames=5
            )
            template = SimpleNamespace(width=12, height=16, scale=1.0)
            first_result = SimpleNamespace(
                detected=True,
                best_match=SimpleNamespace(x=48, y=60, score=0.0, template=template),
            )
            distant_result = SimpleNamespace(
                detected=True,
                best_match=SimpleNamespace(x=4, y=4, score=0.0, template=template),
            )
            frame = np.zeros((100, 96, 3), dtype=np.uint8)

            with mock.patch.object(detector.matcher, "process", side_effect=[first_result] * 4 + [distant_result]):
                for _ in range(5):
                    _, center_x, class_name = detector.process_frame(frame)

            self.assertIsNone(center_x)
            self.assertIsNone(class_name)
            self.assertEqual(len(detector.track_info), 2)
            self.assertEqual(detector.track_info[0].target_history, [True] * 4 + [False])
            self.assertEqual(detector.track_info[1].target_history, [True])

    def test_main_passes_video_frame_to_detector_without_display(self) -> None:
        frame = np.full((10, 15, 3), 127, dtype=np.uint8)
        detector = _FakeDetector("video")
        capture = _FakeCapture(frame)

        with (
            mock.patch.object(detector_module, "CoarseToFineHumanDetector", return_value=detector),
            mock.patch.object(detector_module.cv2, "VideoCapture", return_value=capture) as video_capture,
            mock.patch.object(detector_module.cv2, "imshow") as imshow,
            mock.patch.object(detector_module.cv2, "destroyAllWindows"),
            mock.patch("sys.argv", ["coarse_to_fine_human_detector.py", "--no-display", "--max-frames", "1"]),
        ):
            detector_module.main()

        video_capture.assert_called_once_with("example.mp4")
        self.assertEqual(detector.frames[0][0].shape, (100, 96, 3))
        self.assertEqual(detector.frames[0][1], 1.25)
        self.assertTrue(detector.closed)
        self.assertTrue(capture.released)
        imshow.assert_not_called()

    def test_main_passes_webcam_frame_to_detector_without_timestamp(self) -> None:
        detector = _FakeDetector("camera")
        capture = _FakeCapture(np.zeros((10, 15, 3), dtype=np.uint8))

        with (
            mock.patch.object(detector_module, "CoarseToFineHumanDetector", return_value=detector),
            mock.patch.object(detector_module.cv2, "VideoCapture", return_value=capture) as video_capture,
            mock.patch.object(detector_module.cv2, "destroyAllWindows"),
            mock.patch("sys.argv", ["coarse_to_fine_human_detector.py", "--no-display", "--max-frames", "1"]),
        ):
            detector_module.main()

        video_capture.assert_called_once_with(7)
        self.assertIsNone(detector.frames[0][1])
        self.assertTrue(capture.released)

    def test_main_passes_insight_udp_frame_to_detector(self) -> None:
        detector = _FakeDetector("udp")
        sensor = _FakeInsightSensor(np.zeros((10, 15, 3), dtype=np.uint8))

        with (
            mock.patch.object(detector_module, "CoarseToFineHumanDetector", return_value=detector),
            mock.patch.object(detector_module, "ClsImageViewerUDP", return_value=sensor),
            mock.patch.object(detector_module.cv2, "destroyAllWindows"),
            mock.patch("sys.argv", ["coarse_to_fine_human_detector.py", "--no-display", "--max-frames", "1"]),
        ):
            detector_module.main()

        self.assertEqual(sensor.received, 1)
        self.assertEqual(detector.frames[0][0].shape, (100, 96, 3))
        self.assertIsNone(detector.frames[0][1])
        self.assertTrue(detector.closed)

    def test_main_resizes_only_the_display_frame(self) -> None:
        detector = _FakeDetector("video")
        detector.conf.display_width = 192
        detector.conf.display_height = 200
        capture = _FakeCapture(np.zeros((10, 15, 3), dtype=np.uint8))

        with (
            mock.patch.object(detector_module, "CoarseToFineHumanDetector", return_value=detector),
            mock.patch.object(detector_module.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(detector_module.cv2, "imshow") as imshow,
            mock.patch.object(detector_module.cv2, "waitKey", return_value=-1),
            mock.patch.object(detector_module.cv2, "destroyAllWindows"),
            mock.patch("sys.argv", ["coarse_to_fine_human_detector.py", "--max-frames", "1"]),
        ):
            detector_module.main()

        self.assertEqual(detector.frames[0][0].shape, (100, 96, 3))
        self.assertEqual(imshow.call_args.args[1].shape, (200, 192, 3))
        self.assertEqual(detector.display_sizes, [(192, 200)])

    def test_display_hud_has_a_fixed_font_size_after_resizing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector, template = self._make_detector(Path(temporary_directory), final_max_error=0.1)
            frame = np.zeros((100, 96, 3), dtype=np.uint8)
            frame[60:76, 48:60] = template
            detector.process_frame(frame)

            with mock.patch.object(detector_module.cv2, "putText", wraps=cv2.putText) as put_text:
                display = detector.get_display_frame(192, 200)

            self.assertEqual(display.shape, (200, 192, 3))
            self.assertEqual([call.args[4] for call in put_text.call_args_list], [0.6, 0.55, 0.55])


if __name__ == "__main__":
    unittest.main()
