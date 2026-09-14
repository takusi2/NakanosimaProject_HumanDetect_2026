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

`save_every_n_frames: 1` は全フレームを保存します。容量を抑えたい場合は、値を大きくしてください。

分散フィルタ画像のファイル名末尾は `_variance_filter.png` です。マゼンタ枠は平坦領域フィルタで除外された候補、オレンジ枠は平坦ではないものの参照との相対分散差で除外された候補です。候補枠は重なって表示されます。各フレームJSONの `variance_filters` には、理由別の除外数・候補分散平均・使用した下限値も保存します。詳細照合の枠色は、最終 `MATCH` が緑、最小誤差だが閾値を超えて `NO MATCH` の候補が赤、他の詳細候補が黄です。

検証動画では、ROIを青、最小誤差候補を最終 `MATCH` 時は緑、`NO MATCH` 時は赤で表示します。動画は全処理フレームを保存し、`save_every_n_frames` の画像保存間隔には影響されません。保存を止める場合は、設定に `save_annotated_video: false` を指定してください。
