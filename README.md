# NakanosimaProject_HumanDetect_2026

YOLO による人物検出と OSNet による人物再識別（ReID）を組み合わせ、映像中から**あらかじめ登録した人物**を検出する Python プロジェクトです。

入力には、動画ファイル、PC カメラ、UDP で受信するイメージセンサ画像を利用できます。対象人物が複数フレームで継続して一致した場合に `MATCH` と判定し、その人物の横方向中心座標を出力します。

## 主な機能

- YOLOv11 による人物（COCO class 0）の検出
- OSNet による衣服領域ベースの人物照合
- 参照画像フォルダ内の複数画像との類似度比較
- IoU を用いたフレーム間の簡易追跡と、一定フレーム数での一致確定
- 動画・カメラ・UDP イメージセンサの入力切替
- Center-Surround 形式のセンサ画像から BGR 画像を生成
- `.bin` 形式の記録データを MP4 動画に変換

## 処理の流れ

```text
入力映像
  └─ YOLO: 人物を検出
       └─ 検出枠の衣服領域を切り出し
            └─ OSNet: 特徴量を抽出
                 └─ humanA/ の参照画像と類似度を比較
                      └─ IoU追跡＋複数フレーム判定
                           └─ MATCH / MAYBE / person を描画・出力
```

`MATCH` となった人物のうち、類似度が最も高い人物について `center_x` とクラス名 `A` を標準出力します。

## 構成

```text
.
├─ human_detector.py       # 人物検出・再識別のメインプログラム
├─ ClsImageViewerUDP.py    # UDP画像の受信とCenter-Surround画像の生成
├─ ClsUdpReceiveData.py    # UDP受信の低レベル処理
├─ bin_to_BGRmp4.py        # .binデータから動画を生成するツール
├─ config/
│  ├─ config.yaml          # 動画入力用の設定例
│  └─ config2.yaml         # UDP入力用の設定例（現行の起動設定）
├─ humanA/                 # 対象人物の参照画像（利用者が配置）
├─ bin/                    # センサの記録データ（利用者が配置）
└─ output/                 # 変換済み動画の出力先
```

`humanA/`、`bin/`、`output/`、モデル重みファイルは Git 管理の対象外です。

## セットアップ

Python 3.10 以降を推奨します。仮想環境を作成してから、必要なライブラリをインストールしてください。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision
pip install ultralytics opencv-python pyyaml
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

NVIDIA GPU を使用する場合は、利用する CUDA 環境に対応した PyTorch をインストールしてください。実行時には CUDA が使用可能なら GPU、それ以外では CPU が自動選択されます。

YOLO の重み `yolo11s.pt` は、初回実行時に Ultralytics が取得します。オフライン環境では、あらかじめ重みファイルを配置して、設定の `model_pt` にパスを指定してください。

## 対象人物の参照画像

プロジェクト直下に `humanA/` を作成し、対象人物の画像を 1 枚以上配置します。

```text
humanA/
├─ person_a_front.jpg
├─ person_a_side.jpg
└─ person_a_back.jpg
```

照合では、設定に応じて参照画像の一部（主に衣服領域）を使用します。服装や撮影条件の異なる複数の画像を登録すると、照合の安定化が期待できます。

## 実行

メインプログラムは現在、[`config/config2.yaml`](config/config2.yaml) を読み込む設定です。

```powershell
python human_detector.py
```

表示ウィンドウで `q` を押すと終了します。

### 動画ファイルを入力にする

[`config/config.yaml`](config/config.yaml) を編集し、次のように設定します。

```yaml
input_source: "video"
video_path: "./output/sample_bgr.mp4"
```

`human_detector.py` の末尾にある `conf_path` を `./config/config.yaml` に変更して実行してください。

### PCカメラを入力にする

使用する設定ファイルで以下を指定します。

```yaml
input_source: "camera"
camera_index: 0
```

### UDPイメージセンサを入力にする

[`config/config2.yaml`](config/config2.yaml) は UDP 入力用の例です。

```yaml
input_source: "udp"
frame_width: 160
frame_height: 120
fps: 5
```

UDP 受信部は `127.0.0.1:50002` を使用します。1 セットにつき Center RGB と Surround RGB の計 6 チャンネル（各 160×120、8 bit）を受信し、各チャンネルの Center-Surround 差分を正規化して BGR 画像にします。

## 主な設定項目

| 項目 | 内容 |
| --- | --- |
| `input_source` | `video`、`camera`、`udp` のいずれか |
| `model_pt` | YOLO モデルの重みファイル |
| `pred_thres` | YOLO による人物検出の信頼度閾値 |
| `sample_img_dir` | 対象人物の参照画像フォルダ |
| `reid_model` | 再識別モデル。既定は `osnet_x0_25` |
| `sim_thresh` | 参照画像との類似度閾値 |
| `target_confirm_frames` | `MATCH` 確定に必要な一致フレーム数 |
| `target_confirm_window_frames` | 一致数を数えるフレーム窓の大きさ |
| `match_iou_thresh` | 同一人物として追跡するための IoU 閾値 |
| `min_box_w` / `min_box_h` / `min_box_area` | 再識別の対象とする検出枠の最小サイズ |
| `crop_x_margin` / `crop_y_top` / `crop_y_bottom` | 検出枠から衣服領域を切り出す割合 |

`config.yaml` は動画向け、`config2.yaml` は小さい UDP センサ画像向けに、閾値と最小検出サイズがそれぞれ調整されています。

## 判定表示

| 表示 | 意味 |
| --- | --- |
| `person` | 人物は検出されたが、対象人物とは判定されていない |
| `MAYBE` | そのフレームでは類似度が閾値以上 |
| `MATCH` | 設定したフレーム窓内で必要回数以上一致し、対象人物として確定 |

## `.bin` データから動画を作成する

`bin_to_BGRmp4.py` は `bin/` 内のセンサデータを MP4 に変換します。

Center-Surround 形式では、同じプレフィックスを持つ以下 6 ファイルを 1 セットとして扱います。

```text
<prefix>_0.bin  # Center R
<prefix>_1.bin  # Center G
<prefix>_2.bin  # Center B
<prefix>_3.bin  # Surround R
<prefix>_4.bin  # Surround G
<prefix>_5.bin  # Surround B
```

```powershell
python bin_to_BGRmp4.py
```

変換結果は `output/<prefix>_center_surround.mp4` に出力されます。

## デバッグ出力

設定の `debug_save_gallery_crops` または `debug_save_detect_crops` を `true` にすると、切り出した参照画像・検出画像を `debug_dir` 以下へ保存できます。照合精度を調整する際は、ここで衣服領域が意図どおり切り出されているかを確認してください。

## 注意点

- 参照画像フォルダが空の場合、プログラムはエラーで終了します。
- 人物が小さすぎる場合は、`min_box_*` の条件により再識別の対象外になります。
- UDP 入力はローカルホストで動作する送信側を前提としています。
- リアルタイム性能と精度は、GPU、入力解像度、YOLO モデル、参照画像、各閾値に左右されます。
