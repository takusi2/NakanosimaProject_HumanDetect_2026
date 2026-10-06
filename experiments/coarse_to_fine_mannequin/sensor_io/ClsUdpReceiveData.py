"""UDP画像パケットを受信する小さな補助クラス。

プロジェクト直下の ClsUdpReceiveData.py を、この実験フォルダ単体で
利用できるようにコピーしたもの。
"""

from __future__ import annotations

import os
import socket

import numpy as np

if os.name != "nt":
    import netifaces


class ClsUdpReceiveData:
    """設定された長さのUDP画像パケットを受信する。"""

    def __init__(self) -> None:
        self.sock_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """受信ソケットを閉じる。複数回呼ばれても問題ない。"""
        if getattr(self, "sock_receiver", None) is not None:
            self.sock_receiver.close()
            self.sock_receiver = None

    def check_own_ip(self, interface_name: str) -> str:
        """指定ネットワークインターフェースのIPアドレスを返す。"""
        if os.name == "nt":
            return socket.gethostbyname(socket.gethostname())
        return netifaces.ifaddresses(interface_name)[netifaces.AF_INET][0]["addr"]

    def bind_socket(self, own_ip: str, own_port: int) -> None:
        self.sock_receiver.bind((own_ip, own_port))

    def set_timeout(self, timeout_seconds: float = 0) -> None:
        self.sock_receiver.settimeout(timeout_seconds)

    def set_data_length(
        self, header_length: int, payload_length: int, trailer_length: int
    ) -> None:
        self.header_length = header_length
        self.payload_length = payload_length
        self.trailer_length = trailer_length
        self.data_length = header_length + payload_length + trailer_length

    def receive_data(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """ヘッダ・画素データ・トレーラに分けて1パケットを返す。"""
        try:
            received_buffer, _ = self.sock_receiver.recvfrom(self.data_length)
        except TimeoutError:
            print("Timed out")
            return None, None, None

        payload_end = self.header_length + self.payload_length
        header = np.frombuffer(received_buffer[: self.header_length], dtype=np.uint8)
        payload = np.frombuffer(
            received_buffer[self.header_length : payload_end], dtype=np.uint8
        )
        trailer = np.frombuffer(received_buffer[payload_end : self.data_length], dtype=np.uint8)
        return header, payload, trailer
