"""UDPイメージセンサからCenter-Surround画像を取得する。

プロジェクト直下の ClsImageViewerUDP.py を、この実験フォルダ内だけで
使えるようにしたコピーである。run.py が利用するのは receive_one_set() と
get_center_surround_bgr_image() であり、既存の受信方式は変えていない。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .ClsUdpReceiveData import ClsUdpReceiveData


class ClsImageViewerUDP:
    """160x120・6チャネルのUDPセンサ画像を受信してBGR化する。"""

    def __init__(self) -> None:
        self.set_image_parameter()
        self.set_receiver()
        self.sc_num_of_image = -1
        self.sc_max_num_of_image = 8
        self.sc_max_num_of_proc_image = 8
        self.sc_num_of_proc_image = 1
        self.bl_stop_loop = False
        self.him_received = np.uint8(
            np.zeros((self.sc_max_num_of_image, self.sc_payload_length))
        )
        self.him_processed = np.uint8(
            np.zeros((self.sc_max_num_of_proc_image, self.sc_payload_length))
        )
        self.sc_save_dir = Path("./saved_images")
        self.sc_save_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self) -> None:
        if hasattr(self, "receiver"):
            self.receiver.close()

    def set_image_parameter(self) -> None:
        self.sc_image_width = 160
        self.sc_image_height = 120
        self.sc_payload_length = self.sc_image_width * self.sc_image_height
        self.sc_magnif_rate = 2
        self.sc_blank_width = 5
        self.sc_magnif_image_width = self.sc_image_width * self.sc_magnif_rate
        self.sc_magnif_image_height = self.sc_image_height * self.sc_magnif_rate
        self.sc_num_of_image_x = 4
        self.sc_num_of_image_y = 3
        self.sc_display_width = (
            self.sc_magnif_image_width + self.sc_blank_width
        ) * self.sc_num_of_image_x + self.sc_blank_width
        self.sc_display_height = (
            self.sc_magnif_image_height + self.sc_blank_width
        ) * self.sc_num_of_image_y + self.sc_blank_width
        self.im_display = np.uint8(
            np.zeros((self.sc_display_height, self.sc_display_width))
        )

    def set_receiver(self) -> None:
        self.receiver = ClsUdpReceiveData()
        self.receiver.bind_socket("127.0.0.1", 50002)
        self.receiver.set_data_length(0, self.sc_payload_length, 4)
        self.receiver.set_timeout(3)

    @staticmethod
    def compute_center_surround(
        center_channel: np.ndarray, surround_channel: np.ndarray
    ) -> np.ndarray:
        """Center - Surroundを0〜255へ正規化して返す。"""
        difference = center_channel.astype(np.float32) - surround_channel.astype(np.float32)
        minimum, maximum = difference.min(), difference.max()
        if maximum - minimum <= 1e-6:
            return np.zeros_like(difference, dtype=np.uint8)
        return ((difference - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)

    def receive_one_set(self) -> None:
        """センターRGB・サラウンドRGBの1セットを表示なしで受信する。"""
        if self.sc_num_of_image == -1:
            while True:
                _, _, trailer = self.receiver.receive_data()
                if trailer is not None and trailer[0] == 0:
                    self.sc_num_of_image = int(trailer[1])
                    break

        while True:
            _, payload, trailer = self.receiver.receive_data()
            if payload is None or trailer is None:
                continue
            image_number = int(trailer[0])
            self.him_received[image_number, :] = payload
            if image_number == self.sc_num_of_image - 1:
                break

    def get_image(self, index: int = 0) -> np.ndarray:
        """指定番号の受信画像を160x120の1チャネル画像として返す。"""
        return self.him_received[index, :].reshape(
            (self.sc_image_height, self.sc_image_width)
        )

    def get_center_surround_bgr_image(self) -> np.ndarray:
        """Center RGB・Surround RGBの6画像から差分BGR画像を作る。"""
        if self.sc_num_of_image < 6:
            raise RuntimeError(
                "Center-Surround画像には6チャンネル（Center RGB + Surround RGB）が必要です。"
            )
        center_r, center_g, center_b = (self.get_image(index) for index in range(3))
        surround_r, surround_g, surround_b = (self.get_image(index) for index in range(3, 6))
        return cv2.merge(
            [
                self.compute_center_surround(center_b, surround_b),
                self.compute_center_surround(center_g, surround_g),
                self.compute_center_surround(center_r, surround_r),
            ]
        )
