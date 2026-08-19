# ライブラリ等のインポート ---------------------------------------
import numpy as np
import cv2
from ClsUdpReceiveData import ClsUdpReceiveData
from bin_to_BGRmp4 import compute_center_surround
import os
import datetime


class ClsImageViewerUDP:
    # コンストラクタ --------------------------------------------
    def __init__(self):
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

        # 保存先ディレクトリ ----------------------------------
        self.sc_save_dir = "./saved_images"
        os.makedirs(self.sc_save_dir, exist_ok=True)

    # デストラクタ ----------------------------------------------
    def __del__(self):
        self.receiver.close()

    # 画像変数の設定 -------------------------------------------
    def set_image_parameter(self):
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
        print(self.im_display.shape)

    # 受信クラスの初期化 ----------------------------------------
    def set_receiver(self):
        sc_port_number = 50002
        sc_header_length = 0
        sc_trailer_length = 4
        self.receiver = ClsUdpReceiveData()
        self.receiver.bind_socket("127.0.0.1", sc_port_number)
        self.receiver.set_data_length(
            sc_header_length, self.sc_payload_length, sc_trailer_length
        )
        self.receiver.set_timeout(3)

    # 画像表示 ------------------------------------------------
    def display_image(self):
        for i in range(self.sc_num_of_image):
            im_reshaped = cv2.resize(
                self.him_received[i, :].reshape(
                    (self.sc_image_height, self.sc_image_width)
                ),
                (self.sc_magnif_image_width, self.sc_magnif_image_height),
            )
            sc_row = i // self.sc_num_of_image_x
            sc_col = i % self.sc_num_of_image_x
            sc_row_start = self.sc_blank_width + sc_row * (
                self.sc_magnif_image_height + self.sc_blank_width
            )
            sc_col_start = self.sc_blank_width + sc_col * (
                self.sc_magnif_image_width + self.sc_blank_width
            )
            self.im_display[
                sc_row_start : sc_row_start + self.sc_magnif_image_height,
                sc_col_start : sc_col_start + self.sc_magnif_image_width,
            ] = im_reshaped

    # 処理画像表示 --------------------------------------------
    def display_proc_image(self):
        for i in range(self.sc_num_of_proc_image):
            im_reshaped = cv2.resize(
                self.him_processed[i, :].reshape(
                    (self.sc_image_height, self.sc_image_width)
                ),
                (self.sc_magnif_image_width, self.sc_magnif_image_height),
            )
            sc_row = (i + self.sc_num_of_image) // self.sc_num_of_image_x
            sc_col = (i + self.sc_num_of_image) % self.sc_num_of_image_x
            sc_row_start = self.sc_blank_width + sc_row * (
                self.sc_magnif_image_height + self.sc_blank_width
            )
            sc_col_start = self.sc_blank_width + np.uint16(sc_col) * (
                self.sc_magnif_image_width + self.sc_blank_width
            )
            self.im_display[
                sc_row_start : sc_row_start + self.sc_magnif_image_height,
                sc_col_start : sc_col_start + self.sc_magnif_image_width,
            ] = im_reshaped

        cv2.imshow("Image viewer", self.im_display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.bl_stop_loop = True

    def get_center_surround_bgr_image(self):
        """受信済みのCenter/Surrround各RGB画像から差分BGR画像を作成して返す"""
        if self.sc_num_of_image < 6:
            raise RuntimeError(
                "Center-Surround画像には6チャンネル（Center RGB + Surround RGB）が必要です。"
            )

        center_r = self.get_image(0)
        center_g = self.get_image(1)
        center_b = self.get_image(2)
        surround_r = self.get_image(3)
        surround_g = self.get_image(4)
        surround_b = self.get_image(5)

        diff_r = self.compute_center_surround(center_r, surround_r)
        diff_g = self.compute_center_surround(center_g, surround_g)
        diff_b = self.compute_center_surround(center_b, surround_b)
        return cv2.merge([diff_b, diff_g, diff_r])

    def compute_center_surround(self, center_ch, surround_ch):
        """
        Center-Surround Retinexの計算：
            R = center - surround
        結果を0〜255のuint8にノーマライズして返す。
        """
        c = center_ch.astype(np.float32)
        s = surround_ch.astype(np.float32)
        diff = c - s

        # 0〜255にノーマライズ
        d_min, d_max = diff.min(), diff.max()
        if d_max - d_min > 1e-6:
            diff_norm = (diff - d_min) / (d_max - d_min) * 255.0
        else:
            diff_norm = np.zeros_like(diff)

        return diff_norm.astype(np.uint8)

    # BGR画像の表示・保存 ------------------------------------------
    def display_bgr_image(self):
        im_bgr = self.get_center_surround_bgr_image()
        im_bgr_magnif = cv2.resize(
            im_bgr,
            (self.sc_magnif_image_width, self.sc_magnif_image_height),
        )
        cv2.imshow("BGR merged image", im_bgr_magnif)

        sc_key = cv2.waitKey(1) & 0xFF
        if sc_key == ord("q"):
            self.bl_stop_loop = True
        elif sc_key == ord("s"):
            self.save_bgr_image(im_bgr_magnif)

    # BGR画像をファイル保存 ----------------------------------------
    def save_bgr_image(self, im_bgr):
        st_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        st_filename = os.path.join(self.sc_save_dir, f"bgr_{st_timestamp}.png")
        cv2.imwrite(st_filename, im_bgr)
        print(f"画像を保存しました: {st_filename}")

    # 受信・描画ループ ----------------------------------------
    def receive_and_display(self):
        while True:
            ve_header_rx, ve_payload_rx, ve_trailer_rx = self.receiver.receive_data()

            if self.sc_num_of_image == -1:
                if ve_trailer_rx[0] == 0:
                    self.sc_num_of_image = ve_trailer_rx[1]
            else:
                sc_image_number = ve_trailer_rx[0]
                self.him_received[sc_image_number, :] = ve_payload_rx

                if sc_image_number == self.sc_num_of_image - 1:
                    self.image_processing()
                    self.display_image()
                    # self.display_proc_image()
                    self.display_bgr_image()

            if self.bl_stop_loop:
                break

    # 画像処理 ------------------------------------------------
    def image_processing(self):
        self.him_processed[0, :] = 255 - self.him_received[0, :]

    # 1セット分の画像を受信するだけのメソッド（表示なし） ----------
    def receive_one_set(self):
        # 最初に制御パケット（枚数通知）を受け取る（初回のみ）
        if self.sc_num_of_image == -1:
            while True:
                _, _, ve_trailer_rx = self.receiver.receive_data()
                if ve_trailer_rx[0] == 0:
                    self.sc_num_of_image = ve_trailer_rx[1]
                    break

        # 画像0〜N-1が揃うまで受信し続ける
        while True:
            _, ve_payload_rx, ve_trailer_rx = self.receiver.receive_data()
            sc_image_number = ve_trailer_rx[0]
            self.him_received[sc_image_number, :] = ve_payload_rx
            if sc_image_number == self.sc_num_of_image - 1:
                break

    # 指定番号の画像を(H, W)のグレースケール2次元配列で取得 --------
    def get_image(self, index=0):
        return self.him_received[index, :].reshape(
            (self.sc_image_height, self.sc_image_width)
        )


if __name__ == "__main__":
    viewer = ClsImageViewerUDP()
    viewer.receive_and_display()
