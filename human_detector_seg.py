import os
import time
import glob
import json
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
        model_path = self.conf.seg_model_pt
        self.model = YOLO(model_path).to(self.device)
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

        self.track_info = []  # [{mask, target_history, last_frame_idx}]
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

    def _iou_masks(self, first, second):
        intersection = np.logical_and(first, second).sum()
        union = np.logical_or(first, second).sum()
        return float(intersection / max(union, 1))

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

    def _get_segmentation_data(self, result, detection_index, frame_shape):
        """検出に対応するマスク画像と輪郭点を元画像サイズで返す。"""
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None or detection_index >= len(masks.data):
            return None, None

        H, W = frame_shape[:2]
        mask_data = masks.data[detection_index].detach().cpu().numpy()
        mask = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST) > 0.9
        points = None
        if masks.xy is not None and detection_index < len(masks.xy):
            points = np.asarray(masks.xy[detection_index], dtype=np.float32)
        return mask, points

    def _load_labelme_mask(self, image_path, image_shape):
        """画像と同じbasenameのLabelme JSONから人物マスクを作る。"""
        json_path = os.path.splitext(image_path)[0] + ".json"
        if not os.path.isfile(json_path):
            print(f"Labelme JSONが見つかりません: {json_path}")
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Labelme JSONを読み込めませんでした: {json_path} ({e})")
            return None

        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for shape in annotation.get("shapes", []):
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            if len(points) < 2:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            shape_type = shape.get("shape_type", "polygon")
            if shape_type == "rectangle":
                x1, y1 = np.floor(points[0]).astype(int)
                x2, y2 = np.ceil(points[1]).astype(int)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
            elif shape_type == "polygon":
                polygon = np.round(points).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon], 1)

        if not np.any(mask):
            print(f"Labelme JSONに有効なマスクがありません: {json_path}")
            return None
        return mask.astype(bool)

    def _mask_image_background(self, image, mask):
        background = np.asarray(self.conf.fill_bground_color, dtype=np.uint8)
        return np.where(mask[..., None], image, background).astype(np.uint8)

    def _load_gallery_feats(
        self, sample_img_dir, x_margin, y_top, y_bottom, save_crops=False, save_dir=None
    ):
        paths = sorted(
            path
            for path in glob.glob(os.path.join(sample_img_dir, "*"))
            if os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if save_crops and save_dir is not None:
            os.makedirs(os.path.join(save_dir, "gallery_crops"), exist_ok=True)
        imgs, valid_paths = [], []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                print(f"参照画像を読み込めませんでした: {path}")
                continue
            mask = self._load_labelme_mask(path, img.shape)
            if mask is None:
                continue

            ys, xs = np.where(mask)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
            H, W = img.shape[:2]
            xl, yt, xr, yb = self._crop_clothes(
                x1, y1, x2, y2, H, W, x_margin, y_top, y_bottom
            )
            masked_img = self._mask_image_background(img, mask)
            crop_img = (
                masked_img
                if (xr <= xl or yb <= yt)
                else masked_img[yt:yb, xl:xr]
            )
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
        result = results[0] if len(results) > 0 else None
        masks = getattr(result, "masks", None) if result is not None else None

        crops, metas = [], []
        if masks is not None and masks.data is not None:
            scores = result.boxes.conf if result.boxes is not None else None
            for bi in range(len(masks.data)):
                mask, points = self._get_segmentation_data(result, bi, frame.shape)
                if mask is None or points is None or len(points) < 3:
                    continue

                point_x, point_y = points[:, 0], points[:, 1]
                x1, y1 = int(np.floor(point_x.min())), int(np.floor(point_y.min()))
                x2, y2 = int(np.ceil(point_x.max())), int(np.ceil(point_y.max()))
                H, W = frame.shape[:2]
                x1, x2 = int(np.clip(x1, 0, W - 1)), int(np.clip(x2, 0, W))
                y1, y2 = int(np.clip(y1, 0, H - 1)), int(np.clip(y2, 0, H))

                bw, bh = x2 - x1, y2 - y1
                area = int(mask.sum())
                conf_score = float(scores[bi]) if scores is not None else 0.0
                base_meta = {
                    "rectangle": (x1, y1, x2, y2),
                    "conf": conf_score,
                    "w": bw,
                    "h": bh,
                    "area": area,
                    "seg_area": area,
                    "points": points,
                    "center_point": (
                        float(points[:, 0].mean()), float(points[:, 1].mean())
                    )
                    if points is not None and len(points) > 0
                    else ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "mask": mask,
                }
                # if (
                #     bw < self.conf.min_width
                #     or bh < self.conf.min_height
                #     or area < self.conf.min_area
                # ):
                #     base_meta["skip"] = True
                #     metas.append(base_meta)
                #     continue

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
                if mask is not None:
                    mask_crop = mask[yt:yb, xl:xr]
                    clothes = np.where(
                        mask_crop[..., None], clothes, np.asarray(self.conf.fill_bground_color, dtype=np.uint8)
                    ).astype(np.uint8)
                else:
                    mask_crop = None
                crops.append(clothes)
                base_meta.update(
                    {
                        "skip": False,
                        "crop": clothes,
                        "mask_crop": mask_crop,
                        "bi": bi,
                    }
                )
                metas.append(base_meta)

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
            x1, y1, x2, y2 = m["rectangle"]
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
                        f"[det#{di}] state=SKIP rectangle=({x1},{y1},{x2},{y2}) "
                        f"w={w} h={h} area={area} conf={conf_val} sim=N/A IoU=N/A pos_count=0"
                    )
            else:
                f = features[fi : fi + 1, :]
                fi += 1
                sims = f @ self.Gt  # [1, N]
                sim = float(sims.max())
                is_target_frame = sim >= self.conf.sim_thresh

                # マスクIoUトラッキング
                best_iou, best_idx = 0.0, -1
                for i, track in enumerate(self.track_info):
                    iou = self._iou_masks(m["mask"], track["mask"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                if best_iou >= self.conf.match_mask_iou_thresh:
                    track = self.track_info[best_idx]
                    track["mask"] = m["mask"]
                    track["target_history"].append(is_target_frame)
                    track["target_history"] = track["target_history"][
                        -self.conf.target_confirm_window_frames:
                    ]
                    track["last_frame_idx"] = self.frame_idx
                    updated_track_indices.add(best_idx)
                else:
                    track = {
                        "mask": m["mask"],
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
                    best_match_center_x = m["center_point"][0]

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
                        f"[det#{di}] state={base} rectangle=({x1},{y1},{x2},{y2}) "
                        f"w={w} h={h} area={area} conf={conf_val} sim={sim:.3f} IoU={best_iou:.3f} pos_count={pos}"
                    )

            # セグメンテーション輪郭を描画
            polygon = np.round(m["points"]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [polygon], True, color, 2)
            (_, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
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
    conf_path = "./config/config_seg.yaml"
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