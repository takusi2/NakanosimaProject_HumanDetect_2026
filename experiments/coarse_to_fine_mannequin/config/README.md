# 設定ファイルの使い分け

すべてのYAMLは、次の順番で設定を並べている。

1. 入力設定
2. 表示設定
3. 出力・保存・性能計測設定
4. 検出パラメータ: 粗探索・詳細探索ROI
5. 検出パラメータ: 色特徴量・分散フィルタ
6. 検出パラメータ: 時系列追跡

`run_20260902_1552_right_motor.yaml` は現在の実験条件を記録した設定、`default.yaml` は新しい実験条件を作るための基本設定である。`benchmark_save_on.yaml` と `benchmark_save_off.yaml` は保存処理の速度影響を比較する専用設定であり、通常実行には使わない。

## 引継ぎ時に残すべき設定

検出結果を再現するには、次の項目を必ず残す。

- `template_path`：どの参照画像を使ったか
- `input_source`、使用した `video_path` または `camera_index`：どの入力を使ったか
- `input_width` / `input_height`：検出時の入力サイズ
- `frame_downscale` から `final_max_error`：粗探索・ROI・生検出の条件
- `brightness_weights`、`channel_weights`、`variance_*`、`min_*`：色差と分散フィルタの条件
- `target_confirm_*`、`match_iou_thresh`、`max_track_age`：MATCH確定条件

`device` は検出条件ではないが、CUDA/GPUの有無により実行可否が変わる。引継ぎ時には値を残しつつ、受け取る環境に合わせて `cuda` / `cpu` を確認する。

## 引継ぎ時に必須ではない設定

以下は検出アルゴリズムの結果そのものを決めない、または用途が限定される設定である。

| 項目 | 扱い | 理由 |
| --- | --- | --- |
| `window_name` | 省略候補 | 表示ウィンドウ名だけを変える。 |
| `show_window`、`display_width`、`display_height` | 運用時に決定 | 表示の有無・大きさだけを変える。現在はYAMLとして幅・高さの指定が必要。 |
| `results_root` | 実行環境ごとに変更 | 保存場所だけを変える。 |
| `save` 全体 | デバッグ時だけ詳細指定 | 検出結果ではなく、画像・動画・JSON・CSVを何を保存するかの設定。 |
| `performance` 全体 | 性能評価時だけ必要 | FPS計測と端末表示の設定。検出結果には影響しない。 |
| `processing_frame_limit` | 動作確認時だけ必要 | 先頭から何フレームで止めるかの上限。`null`なら最後まで処理する。 |
| `camera_index` | `input_source: camera` のときだけ必要 | 動画入力・UDP入力では使わない。 |
| `video_path` | `input_source: video` のときだけ必要 | カメラ入力・UDP入力では使わない。 |

ただし、実験結果を完全に再現する目的なら、保存設定・性能計測設定も含めてYAML全体を保存する方が安全である。
