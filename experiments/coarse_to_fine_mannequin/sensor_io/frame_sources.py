"""異なる入力機器を、共通の ``read()`` 操作で扱うための変換役。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .ClsImageViewerUDP import ClsImageViewerUDP


class SensorFrameSource:
    """UDPイメージセンサを、1フレームずつ読む入力元として扱う。"""

    def __init__(self) -> None:
        self._viewer = ClsImageViewerUDP()

    def read(self) -> np.ndarray | None:
        """センサから1セット受信し、センターサラウンド処理済みBGR画像を返す。"""
        self._viewer.receive_one_set()
        return self._viewer.get_center_surround_bgr_image()

    def get_fps(self) -> float:
        """UDP入力は動画ファイルのような固定FPSを持たない。"""
        return 0.0

    def close(self) -> None:
        """UDP受信ソケットを閉じる。"""
        self._viewer.receiver.close()


class CaptureFrameSource:
    """動画ファイルまたはOpenCVカメラを、1フレームずつ読む入力元として扱う。"""

    def __init__(self, source: int | str) -> None:
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open capture input: {source}")

    def read(self) -> np.ndarray | None:
        """動画またはカメラから1フレームを読み、終端時はNoneを返す。"""
        ok, frame = self._capture.read()
        return frame if ok else None

    def get_fps(self) -> float:
        """入力動画のFPSを返す。カメラなどで取得できなければ0を返す。"""
        return float(self._capture.get(cv2.CAP_PROP_FPS))

    def close(self) -> None:
        """動画・カメラのキャプチャを解放する。"""
        self._capture.release()
