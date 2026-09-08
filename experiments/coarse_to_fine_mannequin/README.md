# 粗探索・詳細照合によるマネキン位置推定

縮小フレームで候補領域を絞り、元解像度の候補ROIだけを複数サイズの参照画像で詳細照合する実験です。色特徴量・テンプレート誤差・最小値検索はPyTorch/CUDAで行います。

## 処理の流れ

1. 元参照画像から詳細照合用テンプレートを複数倍率で作り、GPUへ保存する。
2. 実測フレームをGPUへ1回転送し、元解像度特徴量と縮小フレーム特徴量を作る。
3. 縮小フレームを粗探索し、NMSで重複を除いた上位候補ROIを選ぶ。
4. 粗探索座標を元解像度へ戻し、余白を加えたROIを作る。
5. 各ROI内だけで全詳細テンプレートを照合し、全結果で最小誤差の位置を選ぶ。

詳細照合では、テンプレートと候補領域の `RG`・`BY`・明度の局所分散も比較します。対数分散差が `variance_log_distance_max` を超える候補は、色差照合の対象から外れます。均一な白壁への誤検出を減らすための選別です。

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
- `06_scores/`: 全粗探索・詳細照合スコアのCSVとフレームごとのJSON

`save_every_n_frames: 1` は全フレームを保存します。容量を抑えたい場合は、値を大きくしてください。

分散フィルタ画像のファイル名末尾は `_variance_filter.png` です。オレンジの枠は、ROI内で分散閾値を超えて詳細な色差照合から除外された候補領域全体を表します。候補枠は重なって表示されます。詳細照合の枠色は、最終 `MATCH` が緑、最小誤差だが閾値を超えて `NO MATCH` の候補が赤、他の詳細候補が黄です。
