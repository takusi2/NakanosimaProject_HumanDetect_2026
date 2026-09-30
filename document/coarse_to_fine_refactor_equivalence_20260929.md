# 粗探索・詳細探索プログラム分割の同一性確認

## 目的

`experiments/coarse_to_fine_mannequin/src/pipeline.py` を役割別モジュールへ分割しても、検出結果が変わらないことを確認した。

## 実行環境

- 仮想環境: `env`
- Python: 3.13.4
- PyTorch: 2.8.0+cu129
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- 実行設定: `experiments/coarse_to_fine_mannequin/config/refactor_equivalence.yaml`
- 入力動画: `output/tests/2026-0826-1601-05_center_surround.mp4`
- フレーム数: 943

分割前後で、テンプレート、動画、検出設定、CUDA実行環境を同一とした。結果保存先だけを分けた。

## 保存先

- 分割前: `experiments/coarse_to_fine_mannequin/results/refactor_equivalence_before/run_20260929_161111/`
- 分割後: `experiments/coarse_to_fine_mannequin/results/refactor_equivalence_after/run_20260929_161639/`

## 比較結果

| 比較項目 | 分割前 | 分割後 | 結果 |
| --- | ---: | ---: | --- |
| フレームJSON数 | 943 | 943 | 一致 |
| フレームJSONの内容 | - | - | 943件すべてバイト単位で一致 |
| `scores.csv` | - | - | SHA-256が完全一致 |
| 検出結果 | - | - | 座標、テンプレート倍率、スコア、分散フィルタ、時系列MATCH状態まで一致 |

`scores.csv` のSHA-256は、分割前後ともに次の値だった。

```text
A46F78AB21D8322D01CB869BF5A4132A469C40AF6E16978EC0FA45F84478264A
```

## 性能比較

処理時間には実行時の揺らぎがあるが、検出処理の平均時間はほぼ同じだった。

| 指標 | 分割前 | 分割後 |
| --- | ---: | ---: |
| 検出平均時間 | 10.456 ms/frame | 10.450 ms/frame |
| 検出平均FPS | 95.64 | 95.69 |
| 総平均時間 | 11.786 ms/frame | 11.795 ms/frame |
| 総平均FPS | 84.85 | 84.78 |

以上から、今回の変更は検出アルゴリズムを変えない、可読性向上のためのファイル分割であることを確認した。
