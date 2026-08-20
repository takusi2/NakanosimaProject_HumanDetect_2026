import numpy as np
import cv2
import os
import glob
import re

# ============================================================
# パラメータ設定
# ============================================================
BIN_DIR = "./bin/"

# 画像サイズ
IMG_W = 160
IMG_H = 120
PAYLOAD = IMG_W * IMG_H  # 19200

# 出力動画設定
OUTPUT_DIR = "./output/"
OUTPUT_W = 640
OUTPUT_H = 480
OUTPUT_FPS = 110

# ============================================================
# プレフィックスの自動検出（センター・サラウンド用：_0〜_5が全部揃っているセット）
# ============================================================
def find_complete_sets(bin_dir):
    """_0.bin〜_2.binが全て揃っているプレフィックスを返す（BGRセット用）"""
    all_bins = glob.glob(os.path.join(bin_dir, "*_0.bin"))
    complete_sets = []
    for path_0 in sorted(all_bins):
        prefix = re.sub(r"_0\.bin$", "", path_0)
        path_1 = prefix + "_1.bin"
        path_2 = prefix + "_2.bin"
        if os.path.exists(path_1) and os.path.exists(path_2):
            complete_sets.append(os.path.basename(prefix))
        else:
            missing = []
            if not os.path.exists(path_1): missing.append("_1.bin")
            if not os.path.exists(path_2): missing.append("_2.bin")
            print(f"スキップ: {os.path.basename(prefix)} ({', '.join(missing)}が見つかりません)")
    return complete_sets


def find_complete_sets_cs(bin_dir):
    """_0〜_5.binが全て揃っているプレフィックスを返す（センター・サラウンド用）"""
    all_bins = glob.glob(os.path.join(bin_dir, "*_0.bin"))
    complete_sets = []
    for path_0 in sorted(all_bins):
        prefix = re.sub(r"_0\.bin$", "", path_0)
        paths = [prefix + f"_{i}.bin" for i in range(6)]
        missing = [f"_{i}.bin" for i, p in enumerate(paths) if not os.path.exists(p)]
        if not missing:
            complete_sets.append(os.path.basename(prefix))
        else:
            print(f"スキップ: {os.path.basename(prefix)} ({', '.join(missing)}が見つかりません)")
    return complete_sets


# ============================================================
# 1セットの変換処理（既存：BGRのみ）
# ============================================================
def convert_set(bin_dir, prefix, output_dir):
    bin_b = os.path.join(bin_dir, f"{prefix}_2.bin")  # _2 = B
    bin_g = os.path.join(bin_dir, f"{prefix}_1.bin")  # _1 = G
    bin_r = os.path.join(bin_dir, f"{prefix}_0.bin")  # _0 = R

    data_b = np.fromfile(bin_b, dtype=np.uint8)
    data_g = np.fromfile(bin_g, dtype=np.uint8)
    data_r = np.fromfile(bin_r, dtype=np.uint8)

    num_frames_b = len(data_b) // PAYLOAD
    num_frames_g = len(data_g) // PAYLOAD
    num_frames_r = len(data_r) // PAYLOAD

    if not (num_frames_b == num_frames_g == num_frames_r):
        print(f"  警告: フレーム数不一致 B={num_frames_b}, G={num_frames_g}, R={num_frames_r} → 最小に合わせます")
    num_frames = min(num_frames_b, num_frames_g, num_frames_r)
    print(f"  フレーム数: {num_frames}")

    output_path = os.path.join(output_dir, f"{prefix}_bgr.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (OUTPUT_W, OUTPUT_H))

    for i in range(num_frames):
        im_b = data_b[i * PAYLOAD:(i + 1) * PAYLOAD].reshape((IMG_H, IMG_W))
        im_g = data_g[i * PAYLOAD:(i + 1) * PAYLOAD].reshape((IMG_H, IMG_W))
        im_r = data_r[i * PAYLOAD:(i + 1) * PAYLOAD].reshape((IMG_H, IMG_W))

        frame_bgr = cv2.merge([im_b, im_g, im_r])
        frame_resized = cv2.resize(frame_bgr, (OUTPUT_W, OUTPUT_H))
        writer.write(frame_resized)

        if i % 50 == 0:
            print(f"  処理中: {i+1}/{num_frames} フレーム")

    writer.release()
    print(f"  保存完了: {output_path}")


# ============================================================
# センター・サラウンド差分の計算（チャンネルごと）
# ============================================================
def compute_center_surround(center_ch, surround_ch):
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


# ============================================================
# センター・サラウンド差分動画の生成（新関数）
# ============================================================
def convert_set_center_surround(bin_dir, prefix, output_dir):
    """
    _0=センターR, _1=センターG, _2=センターB
    _3=サラウンドR, _4=サラウンドG, _5=サラウンドB
    として読み込み、Center-Surround Retinex差分動画を生成する。
    """
    # センター（_0=R, _1=G, _2=B）
    center_r = np.fromfile(os.path.join(bin_dir, f"{prefix}_0.bin"), dtype=np.uint8)
    center_g = np.fromfile(os.path.join(bin_dir, f"{prefix}_1.bin"), dtype=np.uint8)
    center_b = np.fromfile(os.path.join(bin_dir, f"{prefix}_2.bin"), dtype=np.uint8)

    # サラウンド（_3=R, _4=G, _5=B）
    surround_r = np.fromfile(os.path.join(bin_dir, f"{prefix}_3.bin"), dtype=np.uint8)
    surround_g = np.fromfile(os.path.join(bin_dir, f"{prefix}_4.bin"), dtype=np.uint8)
    surround_b = np.fromfile(os.path.join(bin_dir, f"{prefix}_5.bin"), dtype=np.uint8)

    # フレーム数確認（全チャンネルで一致しているか）
    all_data = {
        "center_R": center_r, "center_G": center_g, "center_B": center_b,
        "surround_R": surround_r, "surround_G": surround_g, "surround_B": surround_b,
    }
    frame_counts = {k: len(v) // PAYLOAD for k, v in all_data.items()}
    print(f"  フレーム数: {frame_counts}")

    if len(set(frame_counts.values())) > 1:
        print("  警告: チャンネル間でフレーム数が不一致 → 最小に合わせます")
    num_frames = min(frame_counts.values())
    print(f"  出力フレーム数: {num_frames}")

    output_path = os.path.join(output_dir, f"{prefix}_center_surround.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (OUTPUT_W, OUTPUT_H))

    for i in range(num_frames):
        # 1フレーム分を取り出してreshape
        def get_frame_ch(data):
            return data[i * PAYLOAD:(i + 1) * PAYLOAD].reshape((IMG_H, IMG_W))

        c_r = get_frame_ch(center_r)
        c_g = get_frame_ch(center_g)
        c_b = get_frame_ch(center_b)
        s_r = get_frame_ch(surround_r)
        s_g = get_frame_ch(surround_g)
        s_b = get_frame_ch(surround_b)

        diff_r = compute_center_surround(c_r, s_r)
        diff_g = compute_center_surround(c_g, s_g)
        diff_b = compute_center_surround(c_b, s_b)

        # 3チャンネルまとめる
        diff_bgr = np.stack([diff_b, diff_g, diff_r], axis=2)  # (H, W, 3)

        frame_bgr = diff_bgr.astype(np.uint8)
        frame_resized = cv2.resize(frame_bgr, (OUTPUT_W, OUTPUT_H))
        writer.write(frame_resized)


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # # ---- 既存：BGR動画変換 ----
    # prefixes_bgr = find_complete_sets(BIN_DIR)
    # if not prefixes_bgr:
    #     print("BGR変換対象のセットが見つかりませんでした。")
    # else:
    #     print(f"\n[BGR変換] {len(prefixes_bgr)}セットを変換します\n")
    #     for i, prefix in enumerate(prefixes_bgr):
    #         print(f"[{i+1}/{len(prefixes_bgr)}] {prefix} を変換中...")
    #         convert_set(BIN_DIR, prefix, OUTPUT_DIR)
    #         print()

    # ---- 新規：Center-Surround差分動画変換 ----
    prefixes_cs = find_complete_sets_cs(BIN_DIR)
    if not prefixes_cs:
        print("Center-Surround変換対象のセットが見つかりませんでした。")
    else:
        print(f"\n[Center-Surround変換] {len(prefixes_cs)}セットを変換します\n")
        for i, prefix in enumerate(prefixes_cs):
            print(f"[{i+1}/{len(prefixes_cs)}] {prefix} を変換中...")
            convert_set_center_surround(BIN_DIR, prefix, OUTPUT_DIR)
            print()

    print("全処理が完了しました。")