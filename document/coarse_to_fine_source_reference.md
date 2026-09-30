# 粗探索・詳細探索プログラム：ファイル・関数リファレンス

この資料は、`experiments/coarse_to_fine_mannequin/` の実行時コードを引き継ぐための説明である。テスト用ファイルは含めない。

## 全体の呼び出し順

```text
run.py: main()
  ├─ ArtifactSaveOptions / PerformanceOptions を設定から作る
  ├─ CoarseToFineMatcher.process(frame)
  │    ├─ features.py        : RG / BY / Brightnessを作る
  │    ├─ coarse_search.py   : 粗探索、NMS、ROI作成
  │    ├─ detail_search.py   : ROI内詳細探索
  │    └─ scoring.py         : 色差と分散フィルタ
  ├─ tracking.py             : MAYBE / MATCHの時系列確定
  ├─ artifacts.py            : 画像、CSV、JSON、検証動画の保存
  └─ performance.py          : 時間・FPSの集計
```

## `run.py`：実行制御

### `_resolve_path(value, config_path)`

```python
path = Path(value)
return path if path.is_absolute() else (config_path.parent / path).resolve()
```

YAML内の相対パスを、YAMLファイルのあるフォルダ基準の絶対パスへ変換する。参照画像、動画、結果フォルダに使う。

### `_create_matcher(template, config)`

```python
return CoarseToFineMatcher(
    template,
    frame_downscale=float(config.get("frame_downscale", 0.25)),
    coarse_template_scales=config.get("coarse_template_scales", [0.2]),
    detail_template_scales=config.get("detail_template_scales", [1.0, 0.8, 0.6, 0.4]),
    ...
)
```

YAMLの検出パラメータを読み、`CoarseToFineMatcher` を1回だけ作る。テンプレート作成もこの時点で行う。

### `_create_tracker(config)`

```python
return TemporalMatchTracker(
    window_frames=int(config.get("target_confirm_window_frames", 10)),
    confirm_frames=int(config.get("target_confirm_frames", 5)),
    match_iou_thresh=float(config.get("match_iou_thresh", 0.8)),
)
```

生検出を時系列で確定する追跡器を作る。ここで作る値が、MATCH確定条件になる。

### `apply_temporal_tracking(raw_result, tracker)`

```python
raw_match = raw_result.detected and best is not None
bbox_xyxy = (best.x, best.y, best.x + best.template.width, best.y + best.template.height)
decision = tracker.update(bbox_xyxy)
```

テンプレート照合の生検出を、追跡器に渡す。`raw_result.detected` は色差閾値まで通過した状態であり、最終的な `FrameResult.detected` は追跡条件も満たしたMATCHだけとなる。

### `_display_frame(...)`

```python
colour = (0, 255, 0) if status == "MATCH" else ...
cv2.rectangle(image, ..., colour, 2)
```

OpenCV表示専用の画像を作る。緑はMATCH、橙はMAYBE、赤はNO MATCH。検出計算には影響しない。

### `_input_size(config)` / `_display_size(config)`

```python
if (width is None) != (height is None):
    raise ValueError("frame_width and frame_height must be specified together")
```

前者は検出前に揃える入力解像度、後者は表示ウィンドウだけの解像度を確認する。片方だけ設定される事故を防ぐ。

### `main()`

```python
raw_result = matcher.process(frame, ...)
result = apply_temporal_tracking(raw_result, tracker)
```

コマンドライン引数を読み、YAML、入力映像、検出器、保存器を準備する主関数。ループ内では「入力取得 → 必要ならリサイズ → 検出 → 追跡 → 保存 → 表示 → 時間計測」を実行する。`finally` 節でカメラ、UDP、動画保存、CSVを必ず閉じる。

## `src/types.py`：結果データの型

このファイルに計算はなく、各段階の結果を同じ形式で受け渡す。

| 型・プロパティ | 内容 |
| --- | --- |
| `Template` | 倍率、BGR画像、GPU特徴量、中心重み、特徴量分散を持つ参照テンプレート |
| `Template.height` / `width` | `features.shape` から高さ・幅を返す |
| `Candidate` | 縮小フレーム上の粗探索候補。テンプレート、左上座標、スコアを持つ |
| `Roi` | 元解像度上の詳細探索領域。元になった粗探索候補も保持する |
| `DetailMatch` | ROI・詳細テンプレートごとの最良候補。座標、色差、分散距離、strideを持つ |
| `DetailVarianceFilter` | 全詳細候補の分散フィルタ結果 |
| `candidate_count` / `passed_count` / `rejected_count` | 分散フィルタ候補数、通過数、除外数 |
| `flatness_rejected_count` | 平坦領域として除外した数。`~self.flatness_passes` を数える |
| `relative_variance_rejected_count` | 平坦ではないが相対分散差で除外した数 |
| `candidate_variance_mean` | RG/BY/Brightness分散の候補全体平均 |
| `FrameResult` | 1フレームの粗探索、ROI、詳細探索、最良候補、最終判定をまとめる |

例えば、候補数を返すプロパティは次の通りである。

```python
@property
def passed_count(self) -> int:
    return int(np.count_nonzero(self.variance_passes))
```

## `src/features.py`：特徴量の作成

### `to_gpu_bgr(image_bgr, device)`

```python
return torch.from_numpy(np.ascontiguousarray(image_bgr)).to(
    device=device, dtype=torch.float32
)
```

OpenCVのBGR配列を、連続メモリの`float32` GPU Tensorへ変換する。元フレームは1フレームにつき1回だけGPUへ送る。

### `normalise_weights(values, device)`

```python
return weights / weights.sum()
```

Brightness計算用のR/G/B重みを正規化する。負値や合計0はエラーにする。

### `normalise_channel_weights(values, device)`

```python
return weights
```

RG・BY・Brightnessの誤差重みを確認してGPUへ置く。こちらは相対的な強さを保つため、合計1には正規化しない。

### `bgr_to_features(image_bgr, brightness_weights)`

```python
rg = red - green
by = (red + green) * 0.5 - blue
brightness = wr * red + wg * green + wb * blue
```

テンプレートと入力領域を、`RG / BY / Brightness` の3特徴量へ変換する。以降の照合と分散はすべてこの3値で行う。

### `center_weight(height, width, min_weight, device)`

```python
return min_weight + (1.0 - min_weight) * (1.0 - torch.clamp(distance, 0.0, 1.0))
```

テンプレート中央が1に近く、端部が`min_weight`に近い重み画像を作る。背景が入りやすい端を過度に重視しないための処理である。

### `downscale_on_gpu(image_bgr, scale)`

```python
return F.interpolate(chw, size=(height, width), mode="area")
```

粗探索用にフレームをGPU上で縮小する。`area`補間で縮小し、CPUへ戻さない。

### `scan_positions(maximum, stride)`

```python
positions = list(range(0, maximum + 1, stride))
if positions[-1] != maximum:
    positions.append(maximum)
```

テンプレート左上の走査位置を作る。最後の端位置も必ず追加するため、右端・下端の候補が探索漏れしない。

## `src/pipeline.py`：検出処理の窓口

### `CoarseToFineMatcher.__init__(...)`

```python
self.coarse_templates = self._create_templates(template_bgr, coarse_template_scales, "coarse")
self.detail_templates = self._create_templates(template_bgr, detail_template_scales, "detail")
```

YAMLから渡されたパラメータを検証し、GPU、重み、粗探索・詳細探索テンプレートを初期化する。フレームごとには呼ばれない。

### `detail_stride_for_scale(scale)`

```python
return max(self.detail_stride_min, int(self.detail_stride_base * scale))
```

倍率ごとの詳細探索strideを返す。小テンプレートほどstrideが小さくなり、位置の見逃しを減らす。

### `_create_templates(original_bgr, scales, prefix)`

```python
image = cv2.resize(original_bgr, (width, height), interpolation=cv2.INTER_AREA)
features = bgr_to_features(tensor, self.brightness_weights)
feature_variance = features.var(dim=(0, 1), correction=0)
```

参照画像を各倍率に縮小し、BGR画像、GPU特徴量、中心重み、RG/BY/Brightnessの分散を`Template`へまとめる。同じ画素サイズになる重複倍率は1つにする。

### `process(frame_bgr, ...)`

```python
coarse_candidates = find_coarse_candidates(...)
rois = make_rois(...)
detail_matches, detail_variance_filters = find_detail_matches(...)
best_match = min(detail_matches, key=lambda item: item.score)
```

1フレームの検出本体。元画像・縮小画像の特徴量を作り、粗探索、ROI作成、詳細探索をこの順に呼ぶ。詳細候補全体のうち色差最小のものを`best_match`とし、`final_max_error`以下なら生検出とする。

## `src/coarse_search.py`：粗探索とROI

### `find_coarse_candidates(...)`

```python
values, indices = torch.topk(candidate_scores.scores.flatten(), k=count, largest=False)
...
available &= distance_squared >= nms_distance_original_px**2
```

縮小フレーム全体に、粗探索用の全テンプレート倍率を適用する。各倍率で低スコア上位候補を取り、元解像度換算の中心間距離でNMSを行う。最後に`coarse_top_k`個だけCPUへ戻す。

### `make_rois(...)`

```python
coarse_uncertainty = coarse_stride / frame_downscale
roi_width = round(max_template_width + 2 * (roi_margin_px + coarse_uncertainty))
```

粗探索候補の中心を元解像度へ戻し、最大詳細テンプレートが入るROIを作る。粗探索strideによる位置ずれも余白へ含め、フレーム外へはみ出さないように切り詰める。

## `src/scoring.py`：色差と分散フィルタ

### `ScoreMap`

`scores`、走査座標、分散距離、平坦領域判定、相対分散判定、候補ごとの分散をまとめるデータ型である。粗探索ではスコアだけ、詳細探索では全項目を使う。

### `score_map(...)`

```python
pixel_error = torch.sqrt(torch.sum(difference.square() * channel_weights, dim=3))
unfiltered_scores = torch.sum(pixel_error * template.weights, dim=(1, 2)) / template.weights.sum()
```

全走査位置の候補領域を一括でGPU抽出し、画素ごとの3特徴量の重み付きユークリッド距離を、中心重み付きで平均する。小さいほど参照画像に近い。

詳細探索での分散フィルタは次の2条件である。

```python
relative_variance_passes = variance_distances <= variance_log_distance_max
flatness_passes = (chroma_variance >= min_chroma_variance) | (
    candidate_variance[:, 2] >= min_brightness_variance
)
variance_passes = relative_variance_passes & flatness_passes
scores = torch.where(variance_passes, unfiltered_scores, float("inf"))
```

`flatness_passes`は色分散と明度分散が両方不足する空・壁を除外する。`relative_variance_passes`は参照テンプレートの分散と大きく異なる候補を除外する。除外候補のスコアは無限大であり、最良候補にはなれない。

### `extract_candidate_patches(...)`

```python
patches = frame_features[
    y_indices[:, None, :, None],
    x_indices[None, :, None, :],
]
```

PythonのループではなくPyTorchの高度インデックスで、全走査位置の候補パッチを一括抽出する。処理速度を保つための重要な関数である。

## `src/detail_search.py`：詳細探索

### `find_detail_matches(...)`

```python
for roi in rois:
    for template in detail_templates:
        stride = detail_stride_for_scale(template.scale)
        candidate_scores = score_map(roi_features, template, stride, ...)
```

各ROIと各詳細テンプレート倍率の組合せを探索する。分散フィルタの可視化またはJSON保存が必要な場合だけ、全候補の判定配列をGPUからCPUへ転送する。

```python
best_index = torch.argmin(flat_scores)
```

組合せごとに最小スコアを1つ選ぶ。すべての候補が分散フィルタで除外されている場合は無限大となり、`DetailMatch`を作らない。

## `src/tracking.py`：時系列MATCH確定

### `TemporalMatchTracker.__init__(...)`

```python
if window_frames <= 0 or confirm_frames <= 0 or confirm_frames > window_frames:
    raise ValueError(...)
```

追跡条件を検証し、トラック一覧、次のID、フレーム番号を初期化する。

### `_iou_xyxy(first, second)`

```python
intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
return intersection / max(first_area + second_area - intersection, 1e-6)
```

2つの矩形のIoUを返す。同じマネキンの連続検出かを判定する指標である。

### `_append_history(track, is_target_frame)`

```python
track.target_history = track.target_history[-self.window_frames :]
```

対象だったかどうかを履歴へ追加し、直近`window_frames`個だけ残す。

### `update(bbox_xyxy)`

```python
if best_iou >= self.match_iou_thresh:
    self._append_history(updated_track, True)
...
confirmed = positive_frames >= self.confirm_frames
```

現在の生検出枠を、IoU最大の既存トラックへ対応付ける。十分なIoUがなければ新規トラックを作る。履歴内の生検出数が`confirm_frames`以上ならMATCH、未満ならMAYBE、生検出なしならNO MATCHとなる。

## `src/artifacts.py`：検証結果の保存

### `ArtifactSaveOptions.from_config(config)`

```python
raw_save = config.get("save", {})
images = nested("images")
scores = nested("scores")
```

YAMLの`save:`設定を読み、保存対象を決める。旧形式の`save_every_n_frames`と`save_annotated_video`も互換のため読み取る。

### `ArtifactSaveOptions.has_frame_output` / `has_any_output`

```python
return self.enabled and any((self.original_frame, ..., self.scores_json))
```

保存対象があるかを返すプロパティ。`save.enabled: false`なら結果フォルダ自体を作らない。

### `ArtifactSaveOptions.should_save_frame(frame_number)`

```python
return self.has_frame_output and frame_number % self.every_n_frames == 0
```

指定間隔のフレームだけを保存するか判定する。

### `ArtifactWriter.__init__(results_root, options)`

```python
self.run_dir = results_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
```

実行IDごとの結果フォルダを作り、CSV保存が有効なら`06_scores/scores.csv`を開く。

### `ArtifactWriter._directory(name)` / `require_run_dir()` / `close()`

```python
directory.mkdir(parents=True, exist_ok=True)
self.csv_file.close()
```

前者は結果サブフォルダを必要時に作る。`require_run_dir`は保存無効時の誤使用を防ぐ。`close`はCSVを閉じる。

### `save_detail_templates(templates)`

```python
template_scale_{template.scale:.3f}.png
```

詳細探索用テンプレート画像を`01_detail_templates/`へ保存する。

### `save_frame(frame_number, original_bgr, result)`

このファイルで最も大きい関数であり、1フレーム分の成果物を保存する。

```python
for rank, candidate in enumerate(result.coarse_candidates, start=1):
    ...
for variance_filter in result.detail_variance_filters:
    ...
for detail in result.detail_matches:
    ...
```

設定に従って、元画像、縮小画像、粗探索候補、ROI、分散除外画像、詳細候補、最良候補、CSV、JSONを保存する。JSONには再現・比較に必要な候補座標、スコア、分散除外数、追跡状態を含める。

### `_write_csv(row)` / `save_performance_summary(summary)`

```python
self.csv_writer.writerow(row)
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
```

前者はCSV出力が有効な場合だけ1行追加する。後者は終了後に性能要約JSONを保存し、フレーム処理時間へ含めない。

### `_draw_box(...)`

```python
cv2.rectangle(image, (x, y), (x + width, y + height), colour, 2)
```

画像に枠とラベルを描く共通関数。画像の上端には黒い背景を置き、文字が読みやすいようにする。

### `_draw_variance_rejections(image, variance_filter)`

```python
relative_rejected = flatness_passes & ~relative_variance_passes
```

詳細ROIに、平坦領域除外をマゼンタ、相対分散差除外を橙で重ねる。候補矩形全体を描くため、重なりが多いほど濃く見える。

### `AnnotatedVideoWriter.__init__` / `write` / `close`

```python
self.writer = cv2.VideoWriter(..., cv2.VideoWriter_fourcc(*"mp4v"), fps, ...)
self.writer.write(draw_annotated_frame(...))
```

`07_annotated_video/coarse_rois_and_best_match.mp4` を作る。`write`が1フレーム追加し、`close`が動画を確定する。

### `draw_annotated_frame(frame_number, original_bgr, result)`

```python
for roi in result.rois:
    cv2.rectangle(image, ..., (255, 0, 0), 1)
```

元フレームへ青のROIと、緑MATCH・橙MAYBE・赤NO MATCHの最良候補を描く。左上にはフレーム番号、状態、スコア、倍率を表示する。

### `_write_image(path, image)`

```python
ok, encoded = cv2.imencode(path.suffix, image)
encoded.tofile(str(path))
```

OpenCVの通常保存で失敗し得る日本語パスにも対応するため、エンコード後に`tofile`で保存する。

## `src/performance.py`：速度計測

### `PerformanceOptions.from_config(config)`

```python
warmup_frames = int(raw.get("warmup_frames", 10))
```

YAMLの`performance:`から、計測の有効化、ウォームアップ除外数、途中表示間隔を読む。

### `PerformanceTracker.__init__(options)`

時間設定と、ウォームアップ後の`FrameTiming`配列を初期化する。

### `record(timing)`

```python
if timing.frame_number > self.options.warmup_frames:
    self._timings.append(timing)
```

初期GPU処理の遅さを性能値へ混ぜないよう、ウォームアップフレームを除外して記録する。

### `should_report(frame_number)`

設定された間隔で途中の性能表示を行うか返す。`0`なら終了時だけ表示する。

### `summary()` / `format_summary()`

```python
"detection": _summarise([item.detection_ms for item in self._timings])
```

入力取得、検出、保存、表示、総時間ごとに統計値を作り、後者は端末表示用の文字列へ整形する。

### `_summarise(milliseconds)`

```python
"mean_fps": 1000.0 / mean_ms
"min_fps": 1000.0 / max(milliseconds)
```

平均・中央値・最小・最大・P95、平均FPS、最も遅いフレーム時のFPSを計算する。

### `_format_stage(name, values)`

```python
f"{name}: mean={values['mean_ms']:.2f}ms ..."
```

1処理区分の統計を、端末へ表示しやすい1行の文字列にする。
