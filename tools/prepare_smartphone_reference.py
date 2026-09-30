"""スマホ写真を、inSightセンサに近い参照画像へ前処理する。

処理順は次の通り。

1. 元のスマホ画像全体へ、RGBチャネルごとに Naka-Rushton を適用する。
2. 人物を含む横長4:3の「疑似センサフレーム」を選び、160x120へ縮小する。
3. 疑似センサフレーム全体へ、Center / Surround の2種類のガウシアンを掛ける。
4. 各B/G/Rチャネルで Center - Surround を計算し、センサ受信側と同じ方法で正規化する。
5. 必要な場合だけ、最後に人物領域を切り出す。

人物だけを先に切り出してからCenter-Surroundを行うと、人物の大きさに応じて
ガウシアンの近傍範囲が変わる。そのため、本ツールでは人物切出しを最後に行う。

注意: FPGAのガウシアン係数そのものは未検証である。center_sigma / surround_sigma
は「設定値をOpenCVのsigmaとして用いる近似」であり、実センサのCenter / Surround
画像と比較して調整する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 利用時にここを書き換える設定
# ---------------------------------------------------------------------------
INPUT_IMAGE_PATH = Path(r"./input/smartphone_reference.jpg")
OUTPUT_DIRECTORY = Path(r"./output/smartphone_reference")

# None の場合、OpenCVの選択画面で疑似センサフレームを選ぶ。
# [x, y, width, height] を設定すれば、同じ領域を再利用できる。
PSEUDO_SENSOR_ROI_XYWH: list[int] | None = None
SELECT_PSEUDO_SENSOR_ROI = True

# Center-Surround後の160x120画像から人物領域を切り出すか。
PERSON_ROI_XYWH: list[int] | None = None
SELECT_PERSON_ROI = False

SENSOR_WIDTH = 160
SENSOR_HEIGHT = 120
CENTER_SIGMA = 1.0
SURROUND_SIGMA = 6.0


def read_bgr_image(path: Path) -> np.ndarray:
    """日本語を含むパスにも対応してBGR画像を読み込む。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読み込めません: {path}")
    return image


def write_bgr_image(path: Path, image_bgr: np.ndarray) -> None:
    """日本語を含むパスにも対応してBGR画像を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image_bgr)
    if not ok:
        raise ValueError(f"画像をエンコードできません: {path}")
    encoded.tofile(str(path))


def cumulative_histogram(channel: np.ndarray) -> np.ndarray:
    """8 bit 1チャネル画像の0-255累積ヒストグラムを返す。"""
    if channel.ndim != 2 or channel.dtype != np.uint8:
        raise ValueError("channel must be a two-dimensional uint8 image")
    return np.cumsum(np.bincount(channel.ravel(), minlength=256), dtype=np.int64)


# Naka-Rushton方程式で圧縮 ###################################################
def Naka_Rushton(
    imImage: np.ndarray, vCumHist: np.ndarray, sHeight: int, sWidth: int
) -> np.ndarray:
    """先輩から共有された式を、ゼロ除算時も安全に使える形で実装する。

    ``imImage`` と ``vCumHist`` の組は、同一の1色チャネルから作る。
    通常の160x120以上の画像では共有版と同じ計算結果になる。
    """
    # 画素数
    sNumOfPixel = sHeight * sWidth
    if sNumOfPixel <= 0:
        raise ValueError("画像サイズは正でなければなりません")
    if vCumHist.shape != (256,):
        raise ValueError("vCumHist must have 256 cumulative values")

    # パラメータ算出
    # alpha
    vIndexForAlpha = np.where(vCumHist >= sNumOfPixel // 256)
    sAlpha = int(vIndexForAlpha[0][0])

    # Ih
    vIndexForIh = np.where(vCumHist >= sNumOfPixel // 2)
    sIh = int(vIndexForIh[0][0])

    # Vmax
    vIndexForVmax = np.where(vCumHist >= sNumOfPixel - (sNumOfPixel // 256))
    sVmaxIndex = int(vIndexForVmax[0][0])
    vmax_denominator = sVmaxIndex - sAlpha
    if vmax_denominator == 0:
        # 一様画像では参照式のVmaxが0除算となる。Center-Surroundも一様領域を
        # 0として扱うため、ここでは出力を0に固定する。
        return np.zeros((sHeight, sWidth), dtype=np.float32)
    sVmax = 255.0 * ((sVmaxIndex - sAlpha) + sIh) / vmax_denominator

    # MM式変換
    numerator = imImage.astype(np.float32) - sAlpha
    denominator = numerator + sIh
    with np.errstate(divide="ignore", invalid="ignore"):
        imMMImage = sVmax * numerator / denominator

    imMMImage = np.nan_to_num(imMMImage, nan=0.0, posinf=255.0, neginf=0.0)
    imMMImage[imMMImage < 0] = 0
    imMMImage[imMMImage > 255] = 255
    return imMMImage


def apply_naka_rushton_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """BGR各チャネルへ、画像全体の累積ヒストグラムを用いてNRを適用する。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 BGR image")

    height, width = image_bgr.shape[:2]
    transformed_channels = []
    for channel in cv2.split(image_bgr):
        transformed = Naka_Rushton(
            channel, cumulative_histogram(channel), height, width
        )
        transformed_channels.append(np.clip(transformed, 0, 255).astype(np.uint8))
    return cv2.merge(transformed_channels)


def validate_roi(roi_xywh: Sequence[int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """[x, y, width, height]形式のROIが画像内にあることを確認する。"""
    if len(roi_xywh) != 4:
        raise ValueError("ROI must contain [x, y, width, height]")
    x, y, width, height = (int(value) for value in roi_xywh)
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError("ROI must be fully inside the image")
    return x, y, width, height


def expand_roi_to_aspect(
    roi_xywh: Sequence[int], image_width: int, image_height: int, target_aspect: float
) -> tuple[int, int, int, int]:
    """選択ROIを含むように拡張し、指定縦横比へ揃える。

    4:3の疑似センサフレームにするために使う。人物を含む選択領域は削らず、
    周囲の背景を可能な範囲で追加する。
    """
    if target_aspect <= 0:
        raise ValueError("target_aspect must be positive")
    x, y, width, height = validate_roi(roi_xywh, image_width, image_height)
    # 選択範囲が縦長なら横だけ、横長なら縦だけを足す。両方をceilで
    # 広げ直すと、整数画素の丸めで必要以上に比率がずれるためである。
    if width / height < target_aspect:
        required_width = int(np.ceil(height * target_aspect))
        required_height = height
    else:
        required_width = width
        required_height = int(np.ceil(width / target_aspect))

    if required_width > image_width or required_height > image_height:
        # 選択範囲を縮小するのではなく、画像内で作れる最大の指定比率フレームを使う。
        required_width = min(image_width, int(np.floor(image_height * target_aspect)))
        required_height = int(round(required_width / target_aspect))
        if required_width < width or required_height < height:
            raise ValueError("選択範囲が大きすぎて、画像内に指定縦横比のフレームを作れません")

    center_x = x + width / 2.0
    center_y = y + height / 2.0
    left = int(round(center_x - required_width / 2.0))
    top = int(round(center_y - required_height / 2.0))
    left = min(max(left, 0), image_width - required_width)
    top = min(max(top, 0), image_height - required_height)
    return left, top, required_width, required_height


def crop_bgr(image_bgr: np.ndarray, roi_xywh: Sequence[int]) -> np.ndarray:
    """ROIを切り出す。"""
    x, y, width, height = validate_roi(roi_xywh, image_bgr.shape[1], image_bgr.shape[0])
    return image_bgr[y : y + height, x : x + width].copy()


def resize_to_sensor_frame(image_bgr: np.ndarray, width: int = SENSOR_WIDTH, height: int = SENSOR_HEIGHT) -> np.ndarray:
    """疑似センサフレームを指定サイズへ縮小・拡大する。"""
    if width <= 0 or height <= 0:
        raise ValueError("出力サイズは正でなければなりません")
    interpolation = cv2.INTER_AREA if image_bgr.shape[1] >= width else cv2.INTER_CUBIC
    return cv2.resize(image_bgr, (width, height), interpolation=interpolation)


def compute_center_surround(center_ch: np.ndarray, surround_ch: np.ndarray) -> np.ndarray:
    """センサ受信側と同じ Center - Surround のチャネル別正規化を行う。"""
    if center_ch.shape != surround_ch.shape:
        raise ValueError("center_ch and surround_ch must have the same shape")
    diff = center_ch.astype(np.float32) - surround_ch.astype(np.float32)
    d_min, d_max = float(diff.min()), float(diff.max())
    if d_max - d_min <= 1e-6:
        return np.zeros_like(center_ch, dtype=np.uint8)
    return ((diff - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)


def apply_center_surround_bgr(
    image_bgr: np.ndarray, *, center_sigma: float, surround_sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像全体へCenter-Surroundを掛け、center/surround/差分BGRを返す。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 BGR image")
    if center_sigma <= 0 or surround_sigma <= 0:
        raise ValueError("ガウシアンsigmaは正でなければなりません")
    if center_sigma >= surround_sigma:
        raise ValueError("center_sigma must be smaller than surround_sigma")

    center_bgr = cv2.GaussianBlur(
        image_bgr, (0, 0), sigmaX=center_sigma, sigmaY=center_sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    surround_bgr = cv2.GaussianBlur(
        image_bgr, (0, 0), sigmaX=surround_sigma, sigmaY=surround_sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    output_channels = [
        compute_center_surround(center_channel, surround_channel)
        for center_channel, surround_channel in zip(
            cv2.split(center_bgr), cv2.split(surround_bgr), strict=True
        )
    ]
    return center_bgr, surround_bgr, cv2.merge(output_channels)


def prepare_sensor_like_frame(
    smartphone_bgr: np.ndarray,
    pseudo_sensor_roi_xywh: Sequence[int],
    *,
    sensor_width: int = SENSOR_WIDTH,
    sensor_height: int = SENSOR_HEIGHT,
    center_sigma: float = CENTER_SIGMA,
    surround_sigma: float = SURROUND_SIGMA,
) -> dict[str, np.ndarray | tuple[int, int, int, int]]:
    """スマホ画像から、Center-Surround済み疑似センサフレームを作る。"""
    naka_rushton_full_bgr = apply_naka_rushton_bgr(smartphone_bgr)
    aspect = sensor_width / sensor_height
    adjusted_roi = expand_roi_to_aspect(
        pseudo_sensor_roi_xywh, smartphone_bgr.shape[1], smartphone_bgr.shape[0], aspect
    )
    naka_rushton_crop_bgr = crop_bgr(naka_rushton_full_bgr, adjusted_roi)
    sensor_frame_nr_bgr = resize_to_sensor_frame(
        naka_rushton_crop_bgr, sensor_width, sensor_height
    )
    center_bgr, surround_bgr, center_surround_bgr = apply_center_surround_bgr(
        sensor_frame_nr_bgr, center_sigma=center_sigma, surround_sigma=surround_sigma
    )
    return {
        "naka_rushton_full_bgr": naka_rushton_full_bgr,
        "naka_rushton_crop_bgr": naka_rushton_crop_bgr,
        "sensor_frame_nr_bgr": sensor_frame_nr_bgr,
        "center_bgr": center_bgr,
        "surround_bgr": surround_bgr,
        "center_surround_bgr": center_surround_bgr,
        "pseudo_sensor_roi_xywh": adjusted_roi,
    }


def select_roi(window_name: str, image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """OpenCV画面でROIを選択し、確定値を返す。"""
    roi = cv2.selectROI(window_name, image_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    if roi[2] <= 0 or roi[3] <= 0:
        raise ValueError("ROIが選択されませんでした")
    return tuple(int(value) for value in roi)


def main() -> None:
    """設定欄に指定したスマホ画像を処理し、中間画像も含めて保存する。"""
    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"INPUT_IMAGE_PATH をスマホ画像へ変更してください: {INPUT_IMAGE_PATH}"
        )
    smartphone_bgr = read_bgr_image(INPUT_IMAGE_PATH)
    selected_roi = PSEUDO_SENSOR_ROI_XYWH
    if selected_roi is None:
        if not SELECT_PSEUDO_SENSOR_ROI:
            raise ValueError("PSEUDO_SENSOR_ROI_XYWHを設定するか、SELECT_PSEUDO_SENSOR_ROIをTrueにしてください")
        selected_roi = select_roi(
            "Select pseudo sensor frame: include the person and surrounding background",
            smartphone_bgr,
        )

    result = prepare_sensor_like_frame(
        smartphone_bgr,
        selected_roi,
        sensor_width=SENSOR_WIDTH,
        sensor_height=SENSOR_HEIGHT,
        center_sigma=CENTER_SIGMA,
        surround_sigma=SURROUND_SIGMA,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_bgr_image(OUTPUT_DIRECTORY / "01_original_smartphone.png", smartphone_bgr)
    write_bgr_image(OUTPUT_DIRECTORY / "02_naka_rushton_full.png", result["naka_rushton_full_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "03_naka_rushton_pseudo_sensor_crop.png", result["naka_rushton_crop_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "04_sensor_frame_naka_rushton.png", result["sensor_frame_nr_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "05_center.png", result["center_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "06_surround.png", result["surround_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "07_center_surround.png", result["center_surround_bgr"])

    person_roi = PERSON_ROI_XYWH
    if person_roi is None and SELECT_PERSON_ROI:
        person_roi = select_roi("Select person after Center-Surround", result["center_surround_bgr"])
    if person_roi is not None:
        person_crop = crop_bgr(result["center_surround_bgr"], person_roi)
        write_bgr_image(OUTPUT_DIRECTORY / "08_person_crop.png", person_crop)

    metadata = {
        "input_image_path": str(INPUT_IMAGE_PATH),
        "processing_order": [
            "Naka-Rushton on full smartphone image per BGR channel",
            "crop pseudo sensor frame and resize to sensor size",
            "Center-Surround on full pseudo sensor frame",
            "optional person crop after Center-Surround",
        ],
        "pseudo_sensor_roi_xywh": result["pseudo_sensor_roi_xywh"],
        "sensor_size": [SENSOR_WIDTH, SENSOR_HEIGHT],
        "center_sigma": CENTER_SIGMA,
        "surround_sigma": SURROUND_SIGMA,
        "hardware_equivalence": "OpenCV GaussianBlur is a software approximation; validate against FPGA output.",
        "person_roi_xywh": person_roi,
    }
    (OUTPUT_DIRECTORY / "processing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"保存完了: {OUTPUT_DIRECTORY}")
    print(f"疑似センサROI [x, y, width, height]: {result['pseudo_sensor_roi_xywh']}")


if __name__ == "__main__":
    main()
