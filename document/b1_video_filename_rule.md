# B-1 マネキン精度検証：動画ファイル名とcfgの規則

この規則は、B-1（マネキン精度検証）で撮影する**入力動画**の名前にだけ適用する。
HumanDetectorの設定は `cfg` として動画とは分けて管理する。

## 基本形式

```text
YYYYMMDD_HHMM_<site>_<lens>_gC<Center>_gS<Surround>_<iir>_<nr>_<light>-<sun_direction>_d<distance>m_<camera_motion>_<scene>_t<trial>.mp4
```

ファイル名には半角英数字、ハイフン（`-`）、アンダースコア（`_`）だけを使う。
小数点は `p` で表記し、すべて小文字で統一する。

## 各項目

| 項目 | 内容 | 表記例 |
| --- | --- | --- |
| `YYYYMMDD_HHMM` | 撮影開始日時 | `20260902_1430` |
| `site` | 撮影場所の識別子 | `2fparking`、`1fsmkarea` |
| `lens` | 広角の有無 | `nowide`、`wide` |
| `gC` | Center Gaussian値 | `gC01` |
| `gS` | Surround Gaussian値 | `gS06` |
| `iir` | IIR設定値または設定ID | `iir040` |
| `nr` | Naka-Rushton設定ID | `nrA` |
| `light` | 明るさの状態 | `shade`、`sun`、`cloudy` |
| `sun_direction` | 直射光の方向 | `none`、`front`、`side`、`back` |
| `d` | センサからマネキンまでの直線距離 | `d4p5m` |
| `camera_motion` | センサ・ロボットの動作状態 | `camstatic`、`camdrive` |
| `scene` | マネキンの状態・向き、または対象なし条件 | 下記参照 |
| `t` | 同一条件での撮影回数 | `t01`、`t02` |

## `scene` の表記

| 条件 | 表記 |
| --- | --- |
| マネキンA・正面 | `manA-front` |
| マネキンA・横向き | `manA-side` |
| マネキンA・斜め向き | `manA-diag` |
| マネキンA・一部遮蔽 | `manA-partial` |
| 対象マネキンなし | `notarget` |
| 対象外人物・別マネキンあり | `distractor` |

## B-1の基準動画例

広角なし、Center:Surround = 1:6、IIR = 0.40、Naka-Rushton設定A、
正面直射、距離4.5 m、カメラ静止、マネキンA正面、1回目の動画は次のように表記する。

```text
20260902_1430_2fparking_nowide_gC01_gS06_iir040_nrA_sun-front_d4p5m_camstatic_manA-front_t01.mp4
```

対象マネキンがいない条件は、次のように表記する。

```text
20260902_1500_2fparking_nowide_gC01_gS06_iir040_nrA_sun-front_d4p5m_camstatic_notarget_t01.mp4
```

ロボット・センサを走行させる条件は、`camera_motion` だけを `camdrive` に変更する。

```text
20260902_1530_2fparking_nowide_gC01_gS06_iir040_nrA_sun-front_d4p5m_camdrive_manA-front_t01.mp4
```

## 運用上の注意

- B-1では、まず `nowide`、`gC01`、`gS06`、`iir040`、`nrA` を基準条件として固定する。
- 比較したい条件だけを変更し、変更しない条件も必ずファイル名に残す。
- 同じ条件を繰り返し撮影する場合は、末尾の `t` だけを増やす。
- 撮影後に動画名を変更する場合も、この規則に従う。

## cfg（HumanDetector設定）の規則

`cfg` は動画の内容ではない。YOLO、OSNet、閾値、追跡、クロップなどの
**HumanDetectorの実行設定一式**を識別するためのIDである。

同じ動画を異なるcfgで実行して精度を比較できるよう、動画名にはcfgを含めない。

```text
同じ入力動画
  ├─ cfgB01 で実行
  ├─ cfgB02 で実行
  └─ cfgB03 で実行
```

### cfg IDと設定ファイル名

B-1の設定IDは、次の形式で統一する。

```text
cfgB<2桁番号>
```

| cfg ID | 設定ファイル | 用途例 |
| --- | --- | --- |
| `cfgB01` | `config/evaluation/cfgB01.yaml` | B-1の基準設定 |
| `cfgB02` | `config/evaluation/cfgB02.yaml` | `cfgB01`から1項目だけ変更した比較設定 |
| `cfgB03` | `config/evaluation/cfgB03.yaml` | 別の比較設定 |

設定番号は2桁で連番にする。欠番は使用してよいが、すでに評価に使った番号を別の内容へ使い回さない。

### cfgに含める項目

cfgには、少なくとも次のHumanDetector設定を保存する。

```text
model_pt
pred_thres
frame_width / frame_height
reid_model
sim_thresh
target_confirm_frames
target_confirm_window_frames
match_iou_thresh
min_box_w / min_box_h / min_box_area
crop_x_margin / crop_y_top / crop_y_bottom
max_track_age
```

`video_path` は入力動画を指定するための値であり、B-1の比較対象となるDetector設定そのものではない。
動画の識別は動画ファイル名と検出結果の記録で行う。

### 新しいcfgを作る手順

1. 基準となるcfgをコピーする。
2. 変更する項目を原則1つに絞る。
3. 新しい番号を付ける。例：`cfgB01` を元にした場合は `cfgB02`。
4. 設定ファイル先頭のコメントに、比較目的と変更点を残す。
5. 一度検出・評価に使ったcfgは内容を変更しない。

設定ファイルの先頭には、次のようなコメントを記録する。

```yaml
# config_id: cfgB02
# purpose: sim_thresh の影響を確認する
# based_on: cfgB01
# changed: sim_thresh 0.65 -> 0.70
```

### B-1での使い方

```text
1. 撮影条件を動画名で記録する
2. 評価したいcfgを1つ選ぶ
3. 同じ動画をそのcfgでHumanDetectorに入力する
4. 検出結果と精度評価結果に、使用cfg IDを記録する
```

この規則により、例えばRecallが変化した場合に、撮影条件の違いなのか、
`sim_thresh` や追跡設定などDetector設定の違いなのかを分けて判断できる。
