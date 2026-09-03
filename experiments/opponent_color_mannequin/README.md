# 二十反対色テンプレート照合によるマネキン位置推定

YOLO と OSNet を使う現行の人物検出とは独立した、マネキン用の実験実装です。参照画像（テンプレート）とカメラフレームを二十反対色・明度へ変換し、重み付きテンプレート照合で位置を推定します。追跡や時系列での確定処理は、この実験の対象外です。色変換、候補領域の誤差計算、最小値検索は PyTorch/CUDA 上で実行します。

## 処理

1. テンプレート BGR 画像から、各画素の `RG`、`BY`、`Y` を計算して保持する。
2. テンプレートと同じ大きさの重みマップを作る。背景とみなす場所の重みを小さくする。
3. フレームを指定ストライドで走査し、テンプレートと同じ大きさの各領域について重み付き誤差を求める。
4. 最小スコアがしきい値以下なら、その領域の中心をマネキン位置として返す。

画素 `(y, x)` の特徴量は、BGR 入力を RGB に読み替えて次で定義します。

```text
RG = R - G
BY = (R + G) / 2 - B
Y  = 0.2126 R + 0.7152 G + 0.0722 B
```

`Y` は Rec.709 の相対輝度です。質問中の `Xr + Yg + Zb / X + Y + Z` は、意図としては一般形の**重み付き平均**、すなわち `(X*r + Y*g + Z*b) / (X + Y + Z)` と考えられます。しかし、重みの根拠や括弧が未確定のため、まずは映像処理で広く使われる Rec.709 を標準値にしています。係数は設定で変更できます。

各候補領域のスコアは、重み付き平均ピクセル距離です。

```text
pixel_error = sqrt(
  w_rg * (RGt - RG)^2 +
  w_by * (BYt - BY)^2 +
  w_y  * (Yt  - Y)^2
)
score = sum(template_weight * pixel_error) / sum(template_weight)
```

重み付き「平均」にしているため、重みマップを変えてもスコアの単位（8 bit の色差）としきい値の比較が保てます。

## 重みマップの候補

`weight_mode` は次の二方式を実装しています。

- `center_falloff`: 中心を 1、端部を `min_weight` に近づける楕円状の連続重み。テンプレート端に少量の背景が含まれる場合に向きます。
- `inner_rectangle`: 上下左右の `margin_*` を完全に 0、内側を 1 にする方式。マネキンの有効領域を矩形で明確に切り出せる場合に向きます。

初回の比較では `inner_rectangle` を推奨します。背景を確実に無視でき、誤検出の原因と重みの効果を切り分けやすいためです。その後、マネキンの輪郭に沿う必要があれば `center_falloff` や将来の手作業マスクを評価します。

## 実行

```powershell
env\Scripts\python.exe -m unittest experiments.opponent_color_mannequin.tests.test_template_matcher
```

実行用の入力・しきい値は [`config/default.yaml`](config/default.yaml) に置いています。動画またはカメラを処理するには、テンプレート画像と入力パスを設定してから次を実行します。

```powershell
env\Scripts\python.exe experiments/opponent_color_mannequin/run.py --config experiments/opponent_color_mannequin/config/default.yaml
```

表示された枠は最小スコアの位置です。緑はしきい値以下（検出）、赤はしきい値超過（未検出）を表し、`q` キーで終了します。

CUDAを使うには、設定へ次を追加できます（省略時も `cuda`）。CUDAが使えない環境では明示的にエラーにします。

```yaml
device: cuda
```

動画の復号、カメラからの受信、OpenCVによる表示だけはCPU側の処理です。OpenCVが受け取った1フレームをGPUへ転送した後の照合処理はGPU上で完結します。

## しきい値の決め方

スコアは色差なので、固定値を先に決め打ちしません。マネキンあり・なしの検証フレーム群で最小スコアを保存し、両者の分布を比較して選びます。照明差が大きい環境では、`channel_weights.y` を下げる、または明度を局所正規化することを次の候補とします。
