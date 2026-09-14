# 粗探索・詳細照合によるマネキン位置推定

縮小フレームで候補領域を絞り、元解像度の候補ROIだけを複数サイズの参照画像で詳細照合する実験です。色特徴量・テンプレート誤差・最小値検索はPyTorch/CUDAで行います。

## 処理の流れ

1. 元参照画像から詳細照合用テンプレートを複数倍率で作り、GPUへ保存する。
2. 実測フレームをGPUへ1回転送し、元解像度特徴量と縮小フレーム特徴量を作る。
3. 縮小フレームを粗探索し、NMSで重複を除いた上位候補ROIを選ぶ。
4. 粗探索座標を元解像度へ戻し、余白を加えたROIを作る。
5. 各ROI内だけで全詳細テンプレートを照合し、全結果で最小誤差の位置を選ぶ。

詳細照合では、テンプレートと候補領域の `RG`・`BY`・明度の局所分散も比較します。対数分散差が `variance_log_distance_max` を超える候補は、色差照合の対象から外れます。均一な白壁への誤検出を減らすための選別です。

さらに、空・白壁・黒壁などの平坦領域を絶対条件で除外します。候補領域の `RG` 分散と `BY` 分散の和が `min_chroma_variance` 未満、かつ Brightness 分散が `min_brightness_variance` 未満なら、相対分散比較・色差計算の前に除外します。どちらか一方に十分な変化があれば候補として残すため、単色の服でも輪郭や影による明度変化を持つマネキンを不必要に除外しにくい設計です。両値を `0.0` にすると、この平坦領域フィルタは無効になります。

詳細探索のstrideはテンプレート倍率ごとに変えます。各倍率を `s` とすると、`max(detail_stride_min, floor(detail_stride_base × s))` です。標準設定の `detail_stride_base: 4`、`detail_stride_min: 2` では、`1.0 → 4`、`0.9 → 3`、`0.8 → 3`、`0.7以下 → 2` になります。小さなテンプレートは細かく探索して見逃しを抑え、大きなテンプレートは探索点を減らして処理時間を抑えます。旧設定の `detail_stride` だけがある設定ファイルも、`detail_stride_base` として読み込めます。

## 実行

設定を [`config/default.yaml`](config/default.yaml) で調整してから、PowerShellで実行します。

```powershell
& .\env\Scripts\python.exe .\experiments\coarse_to_fine_mannequin\run.py --config .\experiments\coarse_to_fine_mannequin\config\default.yaml
```

## 保存される成果物

実行ごとに `results/run_YYYYMMDD_HHMMSS/` が作られ、次を保存します。

- `01_detail_templates/`: 詳細照合用の全テンプレート画像
- `02_frames/`: 元フレームとGPUで生成した縮小フレーム
- `03_coarse_match/`: 縮小フレーム上の粗探索候補
- `04_coarse_rois_original/`: 元解像度へ戻した粗探索ROI
- `05_detail_match/`: ROI・テンプレート倍率ごとの詳細照合結果、分散フィルタで除外された探索位置、最終結果
- `06_scores/`: 全粗探索・詳細照合スコアのCSVとフレームごとのJSON（詳細照合では実際に用いた `stride` も保存）
- `07_annotated_video/coarse_rois_and_best_match.mp4`: 元動画に青の粗探索ROIと、最小誤差候補を重ねた動画

## 保存のON/OFF制御

設定ファイルの `save:` で、検出処理とは独立して保存内容を選べます。`save.enabled: false` にすると、画像・CSV・JSON・検証動画を含めて一切保存せず、検出と表示だけを行います。この場合は `results/run_.../` も作成されません。

```yaml
save:
  enabled: true
  every_n_frames: 5
  detail_templates: false
  images:
    original_frame: false
    coarse_frame: false
    coarse_candidates: false
    coarse_rois: true
    variance_filters: true
    detail_matches: false
    best_match: true
  scores:
    csv: false
    json: true
  annotated_video: true
```

- `every_n_frames`: フレーム画像とCSV/JSONの保存間隔です。`5` なら5フレームごとに保存します。
- `detail_templates`: `01_detail_templates/` の参照テンプレート画像です。
- `images`: 各中間画像を個別に選びます。`variance_filters` は分散フィルタの除外位置画像です。
- `scores.csv` / `scores.json`: 数値結果の保存を個別に選びます。
- `annotated_video`: 青のROIと緑/赤の最終候補を元動画に重ねた検証動画です。これは全処理フレームを保存し、`every_n_frames` の対象外です。

以前の `save_every_n_frames` と `save_annotated_video` だけがある設定ファイルも読み込めますが、今後は `save:` を使用してください。

保存対象ではない縮小フレームや、全詳細候補の分散フィルタ配列はGPUからCPUへ転送しません。なお、分散フィルタによる候補の通過／除外判定そのものは検出に必要なため、GPU上では常に計算します。

## 処理速度の計測

`performance:` を有効にすると、端末にウォームアップ後のフレームごとの時間を集計して表示します。`total` は動画のデコード・検出・保存・表示を合わせた時間、`detection` はGPU照合と候補整理、`artifact_write` は画像・CSV・JSON・検証動画の保存、`display` はOpenCVの表示処理です。各項目について平均・中央値・P95・平均FPS・最低FPSを出力します。

```yaml
performance:
  enabled: true
  warmup_frames: 10
  report_every_n_frames: 0
```

`report_every_n_frames: 0` は終了時だけ、`30` は30フレームごとにも途中結果を表示します。保存が有効で `save.performance_json: true` の場合、計測終了後にだけ `08_performance/performance_summary.json` へ保存します。このJSON書込みはフレーム処理時間の集計後に行うため、比較結果には含まれません。

分散フィルタ画像のファイル名末尾は `_variance_filter.png` です。マゼンタ枠は平坦領域フィルタで除外された候補、オレンジ枠は平坦ではないものの参照との相対分散差で除外された候補です。候補枠は重なって表示されます。各フレームJSONの `variance_filters` には、理由別の除外数・候補分散平均・使用した下限値も保存します。詳細照合の枠色は、最終 `MATCH` が緑、最小誤差だが閾値を超えて `NO MATCH` の候補が赤、他の詳細候補が黄です。

検証動画では、ROIを青、最小誤差候補を最終 `MATCH` 時は緑、`NO MATCH` 時は赤で表示します。
