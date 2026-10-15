"""スマホ写真を、inSightセンサに近い参照画像へ前処理する。

処理順は次の通り。

1. 元のスマホ画像全体へ、RGBチャネルごとに Naka-Rushton を適用する。
2. 上下だけを切り取り、マネキンの高さをセンサ画像に近づける。
3. 縦横比を保ったまま160x120へ収め、余白を黒で埋める。
4. 160x120フレーム全体へ、Center / Surround の2種類のガウシアンを掛ける。
5. 各B/G/Rチャネルで、線形差分と対数差分の両方を計算して保存する。
6. 最終のCenter-Surround画像に対して、必要に応じて彩度を調整する。

スマホ画像内のROIは選択しない。縦長写真では高さを120へ合わせ、横方向に
足りない領域を黒で補う。横長写真では必要に応じて上下を黒で補う。

注意: FPGAのガウシアン係数そのものは未検証である。center_sigma / surround_sigma
は「設定値をOpenCVのsigmaとして用いる近似」であり、実センサのCenter / Surround
画像と比較して調整する。
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 利用時にここを書き換える設定
# ---------------------------------------------------------------------------
INPUT_IMAGE_PATH = Path(r"./humanA/production_enviroment_img2.jpg")
OUTPUT_DIRECTORY = Path(r"./humanA/production_enviroment_img_pre")

# Naka-Rushton後の元画像から、上下を何px取り除くか。
# 横方向は切り取らない。最初は0のまま出力を確認し、マネキンの大きさを
# センサ画像へ近づける必要がある場合だけ値を増やす。
VERTICAL_CROP_TOP_PX = 350
VERTICAL_CROP_BOTTOM_PX = 280

SENSOR_WIDTH = 160
SENSOR_HEIGHT = 120
CENTER_SIGMA = 1.0
SURROUND_SIGMA = 6.0
LOG_EPSILON = 1.0

# 最終参照画像の彩度係数。1.0は変更なし、0.0は完全な無彩色、
# 0.5は彩度を半分にする。まずは1.0のまま、センサ画像との比較を
# 行いながら小さくして調整する。
SATURATION_SCALE = 0.0

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

# ヒストグラム作成##################################################################
def MakeHist(imInput, sLength):

    # 合成画像ヒストグラム配列作成
    vHist, vHist_Bin = np.histogram(imInput, bins=np.arange(0, sLength + 1))

    return vHist


# 累積ヒストグラム配列作成##########################################################
def MakeCumHist(imInput, sLength):

    # 合成画像累積ヒストグラム用配列
    vCumHist = np.zeros((sLength))

    # ヒストグラム作成
    vHist = MakeHist(imInput, sLength)

    # 累積ヒストグラム作成
    for i in range(sLength):
        if(i == 0):
            vCumHist[i] = vHist[i]
        else:
            vCumHist[i] = vHist[i] + vCumHist[i - 1]

    return vCumHist


def MakeCumHist_MM(imInput, sLength, sTargetRedLevel):

    # 合成画像累積ヒストグラム用配列
    vCumHist = np.zeros((sLength))

    # ヒストグラム作成
    vHist = MakeHist(imInput, sLength)

    # 累積ヒストグラム作成
    for i in range(sLength):
        if(i == 0):
            vCumHist[i] = vHist[i]
        else:
            vCumHist[i] = vHist[i] + vCumHist[i - 1]

        if(i == sTargetRedLevel - 1):
            Ih_before = vCumHist[i]
        if(i == sTargetRedLevel):
            Ih = vCumHist[i]

    return vCumHist, Ih_before, Ih



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


def fit_to_sensor_frame_with_black_padding(
    image_bgr: np.ndarray, width: int = SENSOR_WIDTH, height: int = SENSOR_HEIGHT
) -> np.ndarray:
    """画像を切り出さずに160x120へ収め、足りない余白を黒で埋める。"""
    if width <= 0 or height <= 0:
        raise ValueError("出力サイズは正でなければなりません")
    source_height, source_width = image_bgr.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("入力画像サイズは正でなければなりません")
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=interpolation)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    frame[top : top + resized_height, left : left + resized_width] = resized
    return frame


def crop_vertical_region(
    image_bgr: np.ndarray, *, top_pixels: int, bottom_pixels: int
) -> np.ndarray:
    """画像の横幅を保ったまま、上端・下端だけを切り取る。"""
    if top_pixels < 0 or bottom_pixels < 0:
        raise ValueError("上下の切取りpx数は0以上にしてください")
    image_height = image_bgr.shape[0]
    if top_pixels + bottom_pixels >= image_height:
        raise ValueError("上下の切取り後に、少なくとも1pxの高さを残してください")
    end_y = image_height - bottom_pixels if bottom_pixels else image_height
    return image_bgr[top_pixels:end_y, :].copy()


def adjust_saturation_bgr(image_bgr: np.ndarray, saturation_scale: float) -> np.ndarray:
    """BGR画像の彩度だけを係数で調整して返す。

    HSV色空間のS（Saturation: 彩度）だけを ``saturation_scale`` 倍する。
    そのため、H（色相）とV（明るさ）はこの処理では意図的に変更しない。
    ``1.0`` は入力をそのまま返し、``0.0`` は無彩色、``0.0`` より大きい値は
    彩度を強くする。ただし8 bit画像で表せる範囲を超えた値は255へ丸める。

    この調整は、センサ出力と比較する最終のCenter-Surround画像へ行う。
    Naka-RushtonやCenter-Surroundの計算条件を変えずに、参照画像の見た目の
    彩度だけを検証できるようにするためである。
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 BGR image")
    if not np.isfinite(saturation_scale) or saturation_scale < 0:
        raise ValueError("saturation_scale は0以上の有限な値にしてください")
    if saturation_scale == 1.0:
        return image_bgr.copy()

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) * saturation_scale
    hsv[:, :, 1] = np.clip(saturation, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_center_surround(center_ch: np.ndarray, surround_ch: np.ndarray) -> np.ndarray:
    """ログ処理済みCenter/Surroundの差分を、チャネル別に0-255へ正規化する。"""
    if center_ch.shape != surround_ch.shape:
        raise ValueError("center_ch and surround_ch must have the same shape")
    diff = center_ch.astype(np.float32) - surround_ch.astype(np.float32)
    d_min, d_max = float(diff.min()), float(diff.max())
    if d_max - d_min <= 1e-6:
        return np.zeros_like(center_ch, dtype=np.uint8)
    return ((diff - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)


def logarithm_for_center_surround(image: np.ndarray, epsilon: float) -> np.ndarray:
    """CenterまたはSurround画像へ、log(画素値 + epsilon)を適用する。"""
    if epsilon <= 0:
        raise ValueError("log epsilon must be positive")
    return np.log(image.astype(np.float32) + epsilon)


def merge_center_surround_channels(
    center_bgr: np.ndarray, surround_bgr: np.ndarray
) -> np.ndarray:
    """B/G/RごとにCenter-Surround差分を正規化し、BGR画像へ戻す。"""
    center_b, center_g, center_r = cv2.split(center_bgr)
    surround_b, surround_g, surround_r = cv2.split(surround_bgr)

    result_b = compute_center_surround(center_b, surround_b)
    result_g = compute_center_surround(center_g, surround_g)
    result_r = compute_center_surround(center_r, surround_r)
    return cv2.merge([result_b, result_g, result_r])

# 画像の読み込み関数
def Gaussian_func(imInput, sImageHeight, sImageWidth, sNumOfFilter):
    # ## フィルタ用配列準備 ############################################################
    imTempFLOAT = np.zeros((sNumOfFilter * 2 + 1, sImageHeight, sImageWidth))
    imTempFLOAT[0, :, :] = imInput

    # ## フィルタ重み設定 ##############################################################
    sWeight00 = 10
    sWeight01 = 3
    sWeight10 = 8
    sWeight11 = 4

    # ## フィルタ実行 ##################################################################
    for sFilter in range(sNumOfFilter):
        sBaseNumber = int(sFilter / 2) + 1
        if sFilter % 2 == 0:
            sSizeOfFilter = 2 ** sBaseNumber
            vWeight = np.zeros(sSizeOfFilter + 1)
            vWeight[0] = sWeight01
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight00
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight01
        else:
            vWeight[0] = sWeight11
            vWeight[(2 ** (sBaseNumber - 1))] = sWeight10
            vWeight[(2 ** (sBaseNumber - 1)) * 2] = sWeight11

        for sPositionX in range(sImageWidth):
            sCenter = imTempFLOAT[sFilter * 2, :, sPositionX] * vWeight[(2 ** (sBaseNumber - 1))]

            sLatter = imTempFLOAT[sFilter * 2, :, max(0, sPositionX - (2 ** (sBaseNumber - 1)))] * vWeight[0]

            sFormer = imTempFLOAT[sFilter * 2, :, min(sPositionX + (2 ** (sBaseNumber - 1)), sImageWidth - 1)] * vWeight[(2 ** (sBaseNumber - 1)) * 2]

            imTempFLOAT[sFilter * 2 + 1, :, sPositionX] = sFormer + sCenter + sLatter

        imTempFLOAT[sFilter * 2 + 1, :, :] = imTempFLOAT[sFilter * 2 + 1, :, :] / 16

        for sPositionY in range(sImageHeight):
            sCenter = imTempFLOAT[sFilter * 2 + 1, sPositionY, :] * vWeight[(2 ** (sBaseNumber - 1))]

            sLatter = imTempFLOAT[sFilter * 2 + 1, max(sPositionY - (2 ** (sBaseNumber - 1)), 0), :] * vWeight[0]

            sFormer = imTempFLOAT[sFilter * 2 + 1, min(sPositionY + (2 ** (sBaseNumber - 1)), sImageHeight - 1), :] * vWeight[(2 ** (sBaseNumber - 1)) * 2]

            imTempFLOAT[sFilter * 2 + 2, sPositionY, :] = sFormer + sCenter + sLatter

        imTempFLOAT[sFilter * 2 + 2, :, :] = imTempFLOAT[sFilter * 2 + 2, :, :] / 16

    imOutputFLOAT = imTempFLOAT[sNumOfFilter * 2, :, :]

    return imOutputFLOAT


def apply_center_surround_bgr(
    image_bgr: np.ndarray, *, center_sigma: float, surround_sigma: float, log_epsilon: float = LOG_EPSILON
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """画像全体へCenter-Surroundを掛け、線形差分・対数差分の両方を返す。

    FPGA側のログ処理の有無が未確定であるため、次の両方を作る。

    * 線形差分: ``Center - Surround``
    * 対数差分: ``log(Center + epsilon) - log(Surround + epsilon)``
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 BGR image")
    if center_sigma <= 0 or surround_sigma <= 0 or log_epsilon <= 0:
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

    # 研究室のGausian_funcを使う。OpenCVのGaussianBlurはソフトウェア近似であり、FPGAの出力と比較して調整する必要がある。
    # center_b = Gaussian_func(image_bgr[:, :, 0], image_bgr.shape[0], image_bgr.shape[1], int(center_sigma))
    # surround_b = Gaussian_func(image_bgr[:, :, 0], image_bgr.shape[0], image_bgr.shape[1], int(surround_sigma))
    # center_g = Gaussian_func(image_bgr[:, :, 1], image_bgr.shape[0], image_bgr.shape[1], int(center_sigma))
    # surround_g = Gaussian_func(image_bgr[:, :, 1], image_bgr.shape[0], image_bgr.shape[1], int(surround_sigma))
    # center_r = Gaussian_func(image_bgr[:, :, 2], image_bgr.shape[0], image_bgr.shape[1], int(center_sigma))
    # surround_r = Gaussian_func(image_bgr[:, :, 2], image_bgr.shape[0], image_bgr.shape[1], int(surround_sigma))
    # center_bgr = cv2.merge([center_b.astype(np.uint8), center_g.astype(np.uint8), center_r.astype(np.uint8)])
    # surround_bgr = cv2.merge([surround_b.astype(np.uint8), surround_g.astype(np.uint8), surround_r.astype(np.uint8)])

    linear_center_surround_bgr = merge_center_surround_channels(
        center_bgr, surround_bgr
    )
    center_log_bgr = logarithm_for_center_surround(center_bgr, log_epsilon)
    surround_log_bgr = logarithm_for_center_surround(surround_bgr, log_epsilon)
    log_center_surround_bgr = merge_center_surround_channels(
        center_log_bgr, surround_log_bgr
    )
    return center_bgr, surround_bgr, linear_center_surround_bgr, log_center_surround_bgr


def prepare_sensor_like_frame(
    smartphone_bgr: np.ndarray,
    *,
    sensor_width: int = SENSOR_WIDTH,
    sensor_height: int = SENSOR_HEIGHT,
    center_sigma: float = CENTER_SIGMA,
    surround_sigma: float = SURROUND_SIGMA,
    log_epsilon: float = LOG_EPSILON,
    saturation_scale: float = SATURATION_SCALE,
    vertical_crop_top_px: int = VERTICAL_CROP_TOP_PX,
    vertical_crop_bottom_px: int = VERTICAL_CROP_BOTTOM_PX,
) -> dict[str, np.ndarray]:
    """スマホ画像全体から、Center-Surround済みの160x120フレームを作る。"""
    naka_rushton_full_bgr = apply_naka_rushton_bgr(smartphone_bgr)
    naka_rushton_vertical_crop_bgr = crop_vertical_region(
        naka_rushton_full_bgr,
        top_pixels=vertical_crop_top_px,
        bottom_pixels=vertical_crop_bottom_px,
    )
    sensor_frame_nr_bgr = fit_to_sensor_frame_with_black_padding(
        naka_rushton_vertical_crop_bgr, sensor_width, sensor_height
    )
    center_bgr, surround_bgr, linear_center_surround_bgr, log_center_surround_bgr = apply_center_surround_bgr(
        sensor_frame_nr_bgr,
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        log_epsilon=log_epsilon,
    )
    linear_center_surround_saturation_adjusted_bgr = adjust_saturation_bgr(
        linear_center_surround_bgr, saturation_scale
    )
    log_center_surround_saturation_adjusted_bgr = adjust_saturation_bgr(
        log_center_surround_bgr, saturation_scale
    )
    return {
        "naka_rushton_full_bgr": naka_rushton_full_bgr,
        "naka_rushton_vertical_crop_bgr": naka_rushton_vertical_crop_bgr,
        "sensor_frame_nr_bgr": sensor_frame_nr_bgr,
        "center_bgr": center_bgr,
        "surround_bgr": surround_bgr,
        "linear_center_surround_bgr": linear_center_surround_bgr,
        "log_center_surround_bgr": log_center_surround_bgr,
        "linear_center_surround_saturation_adjusted_bgr": (
            linear_center_surround_saturation_adjusted_bgr
        ),
        "log_center_surround_saturation_adjusted_bgr": (
            log_center_surround_saturation_adjusted_bgr
        ),
    }


def main() -> None:
    """設定欄に指定したスマホ画像を処理し、中間画像も含めて保存する。"""
    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"INPUT_IMAGE_PATH をスマホ画像へ変更してください: {INPUT_IMAGE_PATH}"
        )
    smartphone_bgr = read_bgr_image(INPUT_IMAGE_PATH)
    result = prepare_sensor_like_frame(
        smartphone_bgr,
        sensor_width=SENSOR_WIDTH,
        sensor_height=SENSOR_HEIGHT,
        center_sigma=CENTER_SIGMA,
        surround_sigma=SURROUND_SIGMA,
        log_epsilon=LOG_EPSILON,
        saturation_scale=SATURATION_SCALE,
        vertical_crop_top_px=VERTICAL_CROP_TOP_PX,
        vertical_crop_bottom_px=VERTICAL_CROP_BOTTOM_PX,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_bgr_image(OUTPUT_DIRECTORY / "01_original_smartphone.png", smartphone_bgr)
    write_bgr_image(OUTPUT_DIRECTORY / "02_naka_rushton_full.png", result["naka_rushton_full_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "03_naka_rushton_vertical_crop.png", result["naka_rushton_vertical_crop_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "04_sensor_frame_naka_rushton.png", result["sensor_frame_nr_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "05_center.png", result["center_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "06_surround.png", result["surround_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "07_center_surround_linear.png", result["linear_center_surround_bgr"])
    write_bgr_image(OUTPUT_DIRECTORY / "08_center_surround_log.png", result["log_center_surround_bgr"])
    write_bgr_image(
        OUTPUT_DIRECTORY / "09_center_surround_linear_saturation_adjusted.png",
        result["linear_center_surround_saturation_adjusted_bgr"],
    )
    write_bgr_image(
        OUTPUT_DIRECTORY / "10_center_surround_log_saturation_adjusted.png",
        result["log_center_surround_saturation_adjusted_bgr"],
    )

    metadata = {
        "input_image_path": str(INPUT_IMAGE_PATH),
        "processing_order": [
            "Naka-Rushton on full smartphone image per BGR channel",
            "crop top and bottom only from the Naka-Rushton image",
            "fit complete smartphone image into sensor size with black padding",
            "Center-Surround on full 160x120 frame",
            "adjust saturation on each final Center-Surround image",
        ],
        "sensor_size": [SENSOR_WIDTH, SENSOR_HEIGHT],
        "vertical_crop_top_px": VERTICAL_CROP_TOP_PX,
        "vertical_crop_bottom_px": VERTICAL_CROP_BOTTOM_PX,
        "center_sigma": CENTER_SIGMA,
        "surround_sigma": SURROUND_SIGMA,
        "log_epsilon": LOG_EPSILON,
        "saturation_scale": SATURATION_SCALE,
        "center_surround_formulas": {
            "linear": "normalize(center - surround)",
            "log": "normalize(log(center + epsilon) - log(surround + epsilon))",
        },
        "hardware_equivalence": "OpenCV GaussianBlur is a software approximation; validate against FPGA output.",
    }
    (OUTPUT_DIRECTORY / "processing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"保存完了: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
