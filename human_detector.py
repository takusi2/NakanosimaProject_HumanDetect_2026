import os
import time
import glob
from types import SimpleNamespace
import yaml

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from torchreid.reid.utils import compute_model_complexity

from ClsImageViewerUDP import ClsImageViewerUDP

PERSON_CLASS_ID = 0
MARGIN = 2


class HumanDetector:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for key in ("person_color", "maybe_target_color", "confirmed_target_color", "text_color"):
            config[key] = tuple(config[key])

        self.conf = SimpleNamespace(**config)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # 物体検出器/特徴抽出器
        self.model = YOLO(self.conf.model_pt).to(self.device)
        self.model.eval()
        self.extractor = FeatureExtractor(
            model_name=self.conf.reid_model, device=self.device
        )

        if self.conf.print_model_info:
            self.show_model_info()

        # 参照画像から特徴量を取得
        G, gpaths = self._load_gallery_feats(
            self.conf.sample_img_dir,
            x_margin=self.conf.sample_x_margin,
            y_top=self.conf.sample_y_top,
            y_bottom=self.conf.sample_y_bottom,
            save_crops=self.conf.debug_save_gallery_crops,
            save_dir=self.conf.debug_dir,
        )
        if G is None or len(G) == 0:
            raise RuntimeError(f"参照画像が見つかりません: {self.conf.sample_img_dir}")
        print(f"参照画像の特徴次元数: {G.shape}, 参照画像の数: {len(gpaths)}")
        self.G = G
        self.Gt = G.T  # [D, N]

        self.track_info = []  # [{bbox, target_history, last_frame_idx}]
        self.frame_idx = 0

        if self.conf.debug_save_detect_crops:
            os.makedirs(
                os.path.join(self.conf.debug_dir, "detect_crops"), exist_ok=True
            )

    def show_model_info(self):
        print("==== YOLO (Ultralytics) ====")
        try:
            # レイヤ一覧・総パラメータ・GFLOPs など
            self.model.info(verbose=True)
        except Exception as e:
            print(self.model)
            print(f"(yolo.info 失敗: {e})")

        print("\n==== OSNet (torchreid) ====")
        try:
            print(self.extractor.model)  # モデル構造
            # OSNetの複雑度（デフォルト解像度 256x128 相当）
            flops, params = compute_model_complexity(
                self.extractor.model, (1, 3, 256, 128)
            )
            try:
                # 返り値がタプル(フロップス, パラメータ)の数値想定
                print(
                    f"Complexity: FLOPs={float(flops):.2f}G, Params={float(params) / 1e6:.2f}M"
                )
            except Exception:
                # 文字列返却等、フォールバック表示
                print(f"Complexity: {flops}, {params}")
        except Exception as e:
            print(f"OSNetモデル情報の取得に失敗: {e}")

    def _iou_xyxy(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _l2normalize(self, x, axis=1, eps=1e-8):
        if isinstance(x, np.ndarray):
            n = np.linalg.norm(x, axis=axis, keepdims=True)
            return x / np.maximum(n, eps)
        elif torch.is_tensor(x):
            n = torch.linalg.norm(x, dim=axis, keepdim=True)
            n = torch.clamp(n, min=eps)
            return x / n
        return x

    def _extract_feats_bgr_list(self, imgs):
        if len(imgs) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        feats = self.extractor(imgs)  # Tensor or ndarray
        feats = self._l2normalize(feats, axis=1)
        if torch.is_tensor(feats):
            feats = feats.detach().cpu().numpy().astype(np.float32)
        else:
            feats = feats.astype(np.float32)
        return feats

    def _crop_clothes(self, x1, y1, x2, y2, H, W, x_margin, y_top, y_bot):
        w = x2 - x1
        h = y2 - y1
        xl = int(x1 + w * x_margin)
        xr = int(x2 - w * x_margin)
        yt = int(y1 + h * y_top)
        yb = int(y1 + h * y_bot)
        xl = int(np.clip(xl, 0, W - 1))
        xr = int(np.clip(xr, 0, W - 1))
        yt = int(np.clip(yt, 0, H - 1))
        yb = int(np.clip(yb, 0, H - 1))
        return xl, yt, xr, yb

    def _load_gallery_feats(
        self, sample_img_dir, x_margin, y_top, y_bottom, save_crops=False, save_dir=None
    ):
        paths = sorted(glob.glob(os.path.join(sample_img_dir, "*")))
        if save_crops and save_dir is not None:
            os.makedirs(os.path.join(save_dir, "gallery_crops"), exist_ok=True)
        imgs, valid_paths = [], []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                print(f"参照画像を読み込めませんでした: {path}")
                continue
            H, W = img.shape[:2]
            xl, yt, xr, yb = self._crop_clothes(
                0, 0, W, H, H, W, x_margin, y_top, y_bottom
            )
            crop_img = img if (xr <= xl or yb <= yt) else img[yt:yb, xl:xr]
            if save_crops and save_dir is not None:
                base = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(save_dir, "gallery_crops", f"{base}.jpg")
                cv2.imwrite(out, crop_img)
            imgs.append(crop_img)
            valid_paths.append(path)
        if not imgs:
            return None, []
        G = self._extract_feats_bgr_list(imgs)
        print(f"参照画像読込: {len(valid_paths)}枚, feats={G.shape}")
        return G, valid_paths

    # クロップ画像（ReIDに使う服領域）をまとめて表示 ----------------
    def _show_crops(self, crops, window_name="Clothes Crops", crop_disp_size=(128, 256)):
        """crops: [(H, W, 3), ...] のリスト。1枚目の画像のみを表示。空なら黒画像を表示。"""
        
        # リストが空、または1枚目の画像データが不正（Noneやサイズ0）の場合は黒画像を表示
        if not crops or crops[0] is None or crops[0].size == 0:
            blank = np.zeros((crop_disp_size[1], crop_disp_size[0], 3), dtype=np.uint8)
            cv2.imshow(window_name, blank)
            return

        # 1枚目の画像を取得してリサイズ
        first_crop = crops[0]
        resized = cv2.resize(first_crop, crop_disp_size)  # (W, H)

        # 表示
        cv2.imshow(window_name, resized)

    def process_frame(self, frame: np.ndarray):
        """フレーム（描画後）, MATCHのcenter_x, class_name('A' or None)を返す。"""
        self.frame_idx += 1
        t0 = time.time()

        with torch.no_grad():
            results = self.model.predict(
                frame,
                conf=self.conf.pred_thres,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
        boxes = results[0].boxes if len(results) > 0 else None

        crops, metas = [], []
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            for bi, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])

                bw, bh = x2 - x1, y2 - y1
                area = bw * bh
                if (
                    bw < self.conf.min_box_w
                    or bh < self.conf.min_box_h
                    or area < self.conf.min_box_area
                ):
                    metas.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "conf": conf_score,
                            "skip": True,
                            "w": bw,
                            "h": bh,
                            "area": area,
                        }
                    )
                    continue

                # 服の領域をクロップ
                xl, yt, xr, yb = self._crop_clothes(
                    x1,
                    y1,
                    x2,
                    y2,
                    frame.shape[0],
                    frame.shape[1],
                    self.conf.crop_x_margin,
                    self.conf.crop_y_top,
                    self.conf.crop_y_bottom,
                )
                clothes = frame[yt:yb, xl:xr]
                crops.append(clothes)
                metas.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf_score,
                        "skip": False,
                        "crop": clothes,
                        "bi": bi,
                        "w": bw,
                        "h": bh,
                        "area": area,
                    }
                )

        # クロップ画像（服領域）を別ウィンドウで表示
        self._show_crops(crops)     
        print(crops[0].shape if crops else "No crops to show")

        # 特徴抽出
        features = self._extract_feats_bgr_list(crops)

        # ログを1回にまとめる
        det_logs = [] if self.conf.debug_print_detection_logs else None

        best_match_sim = -1.0
        best_match_center_x = None
        updated_track_indices = set()

        # 描画と判定
        fi = 0
        for di, m in enumerate(metas):
            x1, y1, x2, y2 = m["bbox"]
            if m["skip"] or features.shape[0] == 0:
                det_conf = m.get("conf", None)
                label = (
                    f"person: conf={det_conf:.2f}" if det_conf is not None else "person"
                )
                color = self.conf.person_color

                if det_logs is not None:
                    w, h, area = m.get("w"), m.get("h"), m.get("area")
                    conf_val = f"{det_conf:.2f}" if det_conf is not None else "N/A"
                    det_logs.append(
                        f"[det#{di}] state=SKIP bbox=({x1},{y1},{x2},{y2}) "
                        f"w={w} h={h} area={area} conf={conf_val} sim=N/A IoU=N/A pos_count=0"
                    )
            else:
                f = features[fi : fi + 1, :]
                fi += 1
                sims = f @ self.Gt  # [1, N]
                sim = float(sims.max())
                is_target_frame = sim >= self.conf.sim_thresh

                # IOUトラッキング
                best_iou, best_idx = 0.0, -1
                for i, track in enumerate(self.track_info):
                    iou = self._iou_xyxy((x1, y1, x2, y2), track["bbox"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                if best_iou >= self.conf.match_iou_thresh:
                    track = self.track_info[best_idx]
                    track["bbox"] = (x1, y1, x2, y2)
                    track["target_history"].append(is_target_frame)
                    track["target_history"] = track["target_history"][
                        -self.conf.target_confirm_window_frames:
                    ]
                    track["last_frame_idx"] = self.frame_idx
                    updated_track_indices.add(best_idx)
                else:
                    track = {
                        "bbox": (x1, y1, x2, y2),
                        "target_history": [is_target_frame],
                        "last_frame_idx": self.frame_idx,
                    }
                    self.track_info.append(track)
                    updated_track_indices.add(len(self.track_info) - 1)

                pos = sum(track["target_history"])
                is_match = pos >= self.conf.target_confirm_frames
                color = (
                    self.conf.confirmed_target_color
                    if is_match
                    else (
                        self.conf.maybe_target_color
                        if is_target_frame
                        else self.conf.person_color
                    )
                )
                base = (
                    "MATCH" if is_match else ("MAYBE" if is_target_frame else "person")
                )
                if base == "person":
                    det_conf = m.get("conf", None)
                    label = (
                        f"{base}: sim={sim:.2f}, conf={det_conf:.2f}"
                        if det_conf is not None
                        else f"{base}: sim={sim:.2f}"
                    )
                else:
                    label = f"{base}: sim={sim:.2f}"

                # center_x 更新
                if is_match and sim > best_match_sim:
                    best_match_sim = sim
                    best_match_center_x = (x1 + x2) / 2.0

                # 保存（MAYBE/MATCH 時のみ）
                if self.conf.debug_save_detect_crops and base in ("MAYBE", "MATCH"):
                    out_dir = os.path.join(self.conf.debug_dir, "detect_crops")
                    os.makedirs(out_dir, exist_ok=True)
                    crop_img = m.get("crop")
                    bi = m.get("bi", 0)
                    fname = f"f{self.frame_idx:06d}_{base}_sim{sim:.2f}_i{bi}.jpg"
                    cv2.imwrite(os.path.join(out_dir, fname), crop_img)

                # 検出ごとのログ（後でまとめて出力）
                if det_logs is not None:
                    w, h, area = m.get("w"), m.get("h"), m.get("area")
                    det_conf = m.get("conf", None)
                    conf_val = f"{det_conf:.2f}" if det_conf is not None else "N/A"
                    det_logs.append(
                        f"[det#{di}] state={base} bbox=({x1},{y1},{x2},{y2}) "
                        f"w={w} h={h} area={area} conf={conf_val} sim={sim:.3f} IoU={best_iou:.3f} pos_count={pos}"
                    )

            # 描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, max(y1 - text_h - MARGIN, 0)),
                (x1 + text_w + MARGIN, max(y1 + MARGIN, text_h + MARGIN)),
                color,
                thickness=-1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, max(y1, text_h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.conf.text_color,
                1,
                cv2.LINE_AA,
            )

        for track_index, track in enumerate(self.track_info):
            if track_index not in updated_track_indices:
                track["target_history"].append(False)
                track["target_history"] = track["target_history"][
                    -self.conf.target_confirm_window_frames:
                ]

        # 古いトラックを削除
        self.track_info = [
            t
            for t in self.track_info
            if (self.frame_idx - t["last_frame_idx"]) <= self.conf.max_track_age
        ]

        # FPS表示
        elapsed = time.time() - t0
        fps = 1.0 / max(elapsed, 1e-6)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # ログ出力
        header = f"Frame {self.frame_idx}: 検出人数={len(metas)}, clothes_feature_shape={features.shape}, 時間={elapsed:.3f}s"
        if det_logs is not None and len(det_logs) > 0:
            print("\n".join([header] + det_logs))
        else:
            print(header)

        # 返り値: 描画後フレーム, MATCHのcenter_x, class_name('A' or None)
        class_name = "A" if best_match_center_x is not None else None
        center_x = (
            float(best_match_center_x) if best_match_center_x is not None else None
        )
        return frame, center_x, class_name

if __name__ == "__main__":
    conf_path = "./config/config2.yaml"
    model = HumanDetector(config_path=conf_path)

    INPUT_SOURCE = model.conf.input_source
    VIDEO_PATH   = model.conf.video_path
    CAMERA_INDEX = model.conf.camera_index

    if INPUT_SOURCE == "udp":
        sensor = ClsImageViewerUDP()
        def get_frame():
            sensor.receive_one_set()
            frame = sensor.get_center_surround_bgr_image()
            if frame is None:
                return False, None
            frame = cv2.resize(frame, (model.conf.frame_width, model.conf.frame_height))
            return True, frame

    elif INPUT_SOURCE == "camera":
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"カメラ(index={CAMERA_INDEX})を開けませんでした。")
        def get_frame():
            ret, frame = cap.read()
            if not ret:
                return False, None
            frame = cv2.resize(frame, (model.conf.frame_width, model.conf.frame_height))
            return True, frame

    elif INPUT_SOURCE == "video":
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"動画ファイルを開けませんでした: {VIDEO_PATH}")
        def get_frame():
            ret, frame = cap.read()
            if not ret:
                return False, None
            frame = cv2.resize(frame, (model.conf.frame_width, model.conf.frame_height))
            return True, frame

    else:
        raise ValueError(f"不明なINPUT_SOURCE: '{INPUT_SOURCE}'")

    try:
        while True:
            ret, frame = get_frame()
            if not ret:
                print("入力終了。" if INPUT_SOURCE == "video" else "画像を取得できませんでした。終了します。")
                break
            out_frame, cx, cname = model.process_frame(frame)
            print(f"中心座標: {cx}, クラス: {cname}")
            cv2.imshow("Human ReID Detector", out_frame)
            if cx is not None:
                print(f"MATCH center_x={cx:.1f}, class={cname}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if INPUT_SOURCE in ("camera", "video"):
            cap.release()
        cv2.destroyAllWindows()