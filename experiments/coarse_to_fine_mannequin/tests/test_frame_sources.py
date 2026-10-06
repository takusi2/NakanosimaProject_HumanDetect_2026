"""入力元を共通のread()操作へ変換する処理を確認する。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

from sensor_io.frame_sources import (
    CaptureFrameSource,
    SensorFrameSource,
)


class FrameSourceTests(unittest.TestCase):
    def test_sensor_source_reads_processed_sensor_frame(self) -> None:
        expected_frame = np.full((4, 6, 3), 100, dtype=np.uint8)
        viewer = mock.Mock()
        viewer.get_center_surround_bgr_image.return_value = expected_frame

        with mock.patch("sensor_io.frame_sources.ClsImageViewerUDP", return_value=viewer):
            source = SensorFrameSource()
            frame = source.read()
            source.close()

        self.assertIs(frame, expected_frame)
        viewer.receive_one_set.assert_called_once_with()
        viewer.receiver.close.assert_called_once_with()
        self.assertEqual(source.get_fps(), 0.0)

    def test_capture_source_returns_frame_and_releases_capture(self) -> None:
        expected_frame = np.full((4, 6, 3), 100, dtype=np.uint8)
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = True, expected_frame
        capture.get.return_value = 29.97

        with mock.patch("sensor_io.frame_sources.cv2.VideoCapture", return_value=capture):
            source = CaptureFrameSource("input.mp4")
            frame = source.read()
            source.close()

        self.assertIs(frame, expected_frame)
        self.assertEqual(source.get_fps(), 29.97)
        capture.release.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
