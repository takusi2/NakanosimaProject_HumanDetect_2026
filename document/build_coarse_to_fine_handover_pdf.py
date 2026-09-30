"""粗探索・詳細探索マネキン検出プログラムの引継ぎPDFを生成する。"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "coarse_to_fine_mannequin_handover.pdf"

pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
JP_FONT = "HeiseiKakuGo-W5"
CODE_FONT = "Courier"

NAVY = colors.HexColor("#19324D")
BLUE = colors.HexColor("#EAF4FF")
BLUE_BORDER = colors.HexColor("#2C7BB6")
GREEN = colors.HexColor("#EAF7EE")
GREEN_BORDER = colors.HexColor("#2E8B57")
RED = colors.HexColor("#FFF0F0")
RED_BORDER = colors.HexColor("#BF3F3F")
GRAY = colors.HexColor("#F4F6F8")
TEXT = colors.HexColor("#202A33")


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleJP", parent=base["Title"], fontName=JP_FONT, fontSize=25,
            leading=34, alignment=TA_CENTER, textColor=NAVY, spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleJP", parent=base["Normal"], fontName=JP_FONT, fontSize=11,
            leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#51606F"),
        ),
        "h1": ParagraphStyle(
            "H1JP", parent=base["Heading1"], fontName=JP_FONT, fontSize=18,
            leading=25, textColor=NAVY, spaceBefore=4, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2JP", parent=base["Heading2"], fontName=JP_FONT, fontSize=14,
            leading=20, textColor=NAVY, spaceBefore=11, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3JP", parent=base["Heading3"], fontName=JP_FONT, fontSize=12,
            leading=17, textColor=TEXT, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyJP", parent=base["BodyText"], fontName=JP_FONT, fontSize=9.3,
            leading=15, textColor=TEXT, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallJP", parent=base["BodyText"], fontName=JP_FONT, fontSize=8,
            leading=12, textColor=TEXT,
        ),
        "caption": ParagraphStyle(
            "CaptionJP", parent=base["BodyText"], fontName=JP_FONT, fontSize=8.2,
            leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#4F5B66"),
        ),
        "box_title": ParagraphStyle(
            "BoxTitle", parent=base["BodyText"], fontName=JP_FONT, fontSize=9.3,
            leading=13, textColor=NAVY,
        ),
    }


S = make_styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [P(f"• {item}") for item in items]


def section(title: str) -> list:
    return [Paragraph(title, S["h1"])]


def subsection(title: str) -> list:
    return [Paragraph(title, S["h2"])]


def signature_box(kind: str, signature: str, explanation: str, *, red: bool = False) -> Table:
    background, border = (RED, RED_BORDER) if red else (GREEN, GREEN_BORDER)
    header = f"<b>{kind}</b>  {signature}"
    table = Table(
        [[P(header, "box_title")], [Preformatted(signature, ParagraphStyle("Code", fontName=CODE_FONT, fontSize=7.3, leading=9.2))], [P(explanation, "small")]],
        colWidths=[174 * mm],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), background),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, border),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def info_box(title: str, lines: list[str]) -> Table:
    body = "<br/>".join(f"• {line}" for line in lines)
    table = Table([[P(f"<b>{title}</b>", "box_title")], [P(body, "small")]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLUE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def flow_box(label: str, detail: str, color: colors.Color = BLUE, border: colors.Color = BLUE_BORDER) -> Table:
    table = Table([[P(f"<b>{label}</b><br/>{detail}", "small")]], colWidths=[33 * mm], rowHeights=[19 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def arrow() -> Paragraph:
    return P("→", "h2")


def add_function(story: list, name: str, code: str, purpose: str, points: list[str], *, is_class: bool = False) -> None:
    story.append(Paragraph(name, S["h3"]))
    story.append(signature_box("class" if is_class else "method", code, purpose, red=is_class))
    story.extend(bullets(points))
    story.append(Spacer(1, 3))


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#B5C1CC"))
        canvas.line(18 * mm, 286 * mm, 192 * mm, 286 * mm)
        canvas.setFont(JP_FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#5B6670"))
        canvas.drawString(18 * mm, 290 * mm, "粗探索・詳細探索テンプレートマッチングによるマネキン検出プログラム 引継ぎ資料")
        canvas.drawRightString(192 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=18 * mm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])
    story: list = []

    # Cover
    story.extend([Spacer(1, 42 * mm), Paragraph("粗探索・詳細探索テンプレートマッチングによる", S["title"]), Paragraph("マネキン検出プログラム", S["title"]), Spacer(1, 9 * mm)])
    story.append(info_box("資料の目的", [
        "experiments/coarse_to_fine_mannequin/ の検出処理を、初めて読む人でも追えるように説明する。",
        "処理の順序、ファイルの役割、主要なクラス・関数、設定値と保存結果の関係をまとめる。",
        "実験用プログラムの引継ぎ資料であり、ロボット搭載版へ移す際の参照資料としても使う。",
    ]))
    story.extend([Spacer(1, 15 * mm), P("作成日: 2026年9月29日", "subtitle"), P("対象ブランチ: codex/refactor-coarse-to-fine-modules", "subtitle"), Spacer(1, 12 * mm)])
    story.append(P("対象フォルダ", "h2"))
    story.append(info_box("experiments/coarse_to_fine_mannequin/", [
        "run.py: 実行入口。動画入力、検出、追跡、保存、表示を統括する。",
        "src/: 処理を役割ごとに分けた検出ロジック。",
        "config/: YAML形式の検出設定。",
        "results/: 実行時に生成される検証結果。",
    ]))
    story.append(PageBreak())

    # Overview
    story += section("1. 全体像")
    story.append(P("この手法は、最初から元解像度の全画面を細かく探索しない。まず縮小フレームでマネキンがありそうな領域を粗く選び、その周辺だけを元解像度で細かく照合する。これにより、精度を保ちながら計算量を抑える。"))
    story.append(Spacer(1, 3))
    flow = [[
        flow_box("入力", "動画フレーム\n参照画像"), arrow(),
        flow_box("特徴量", "BGR → RG / BY / Brightness"), arrow(),
        flow_box("粗探索", "縮小画面で\n候補を抽出"), arrow(),
        flow_box("ROI", "元解像度の\n詳細領域を作成"),
    ], [
        flow_box("詳細探索", "ROI内を全倍率で\nテンプレート照合", GREEN, GREEN_BORDER), arrow(),
        flow_box("分散フィルタ", "平坦領域・\n分散差を除外", GREEN, GREEN_BORDER), arrow(),
        flow_box("追跡", "MAYBE / MATCHを\n時系列で確定", RED, RED_BORDER), arrow(),
        flow_box("保存・表示", "動画 / CSV / JSON\n性能統計"),
    ]]
    tbl = Table(flow, colWidths=[33*mm, 7*mm, 33*mm, 7*mm, 33*mm, 7*mm, 33*mm], hAlign="CENTER")
    tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [tbl, Spacer(1, 3), P("図1. 1フレームに対する処理の流れ。緑は詳細照合時の選別、赤はフレーム間での確定処理。", "caption")]
    story += subsection("1.1 ファイル構成")
    data = [[P("ファイル", "small"), P("担当する処理", "small")]]
    for file, role in [
        ("run.py", "設定読込、入力、検出・追跡・保存の実行制御"),
        ("src/pipeline.py", "粗探索から詳細探索までを順番に呼ぶ検出器の窓口"),
        ("src/features.py", "特徴量化、GPU転送、縮小、中心重み"),
        ("src/types.py", "各段階の結果を保持するデータ型"),
        ("src/coarse_search.py", "粗探索、NMS、ROI生成"),
        ("src/detail_search.py", "ROI内の詳細テンプレート照合"),
        ("src/scoring.py", "色差と分散フィルタ"),
        ("src/tracking.py", "IoUでの時系列MATCH確定"),
        ("src/artifacts.py", "画像、CSV、JSON、検証動画の保存"),
        ("src/performance.py", "時間・FPSの集計"),
    ]:
        data.append([P(f"<font name='{CODE_FONT}'>{file}</font>", "small"), P(role, "small")])
    table = Table(data, colWidths=[55*mm, 119*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), NAVY), ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("GRID", (0,0),(-1,-1),0.3,colors.HexColor("#B8C1C8")),
        ("BACKGROUND", (0,1),(-1,-1), colors.white),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, GRAY]),
        ("VALIGN", (0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story += [table, PageBreak()]

    # run.py
    story += section("2. 実行制御: run.py")
    story.append(P("run.py は実験を開始する入口である。検出の計算式はここに置かず、設定読込・入力・表示・保存・追跡の順序を管理する。まずこのファイルと pipeline.py を読むと、全体のつながりを理解しやすい。"))
    add_function(story, "main()", "def main() -> None:", "プログラムの主関数。設定を読み、動画の各フレームに対して検出と後処理を行う。", [
        "処理順は「入力取得 → 必要ならリサイズ → matcher.process → 時系列追跡 → 保存 → 表示 → 性能記録」。",
        "finally 節で VideoCapture、UDP、保存動画、CSV を閉じるため、途中で止めてもファイルが壊れにくい。",
    ])
    add_function(story, "_resolve_path(value, config_path)", "def _resolve_path(value, config_path) -> Path:", "YAML内の相対パスを、YAMLファイルが置かれたフォルダ基準で絶対パスへ変換する。", [
        "テンプレート画像・入力動画・results_root に使う。", "PCや実行場所が変わっても設定ファイルの移植性を保つ。",
    ])
    add_function(story, "_create_matcher(template, config)", "def _create_matcher(template, config) -> CoarseToFineMatcher:", "YAMLの検出パラメータから CoarseToFineMatcher を1回だけ作る。", [
        "テンプレートの倍率別特徴量も初期化時にGPUへ置くため、毎フレーム作り直さない。",
        "frame_downscale、テンプレート倍率、stride、分散フィルタなどはここから渡される。",
    ])
    add_function(story, "_create_tracker(config)", "def _create_tracker(config) -> TemporalMatchTracker:", "target_confirm_window_frames、target_confirm_frames、match_iou_thresh を読み、追跡器を作る。", [
        "テンプレート照合の生結果と、最終MATCHは異なる。最終判断は tracker が行う。",
    ])
    add_function(story, "apply_temporal_tracking(raw_result, tracker)", "def apply_temporal_tracking(raw_result, tracker) -> FrameResult:", "生検出の最良候補を追跡器へ渡し、MATCH / MAYBE / NO MATCH を FrameResult に反映する。", [
        "raw_result.detected は色差が final_max_error 以下かどうか。", "最終 detected は指定窓内で指定回数以上の生検出が続いたMATCHだけ。",
    ])
    add_function(story, "_display_frame / _input_size / _display_size", "def _display_frame(...):\ndef _input_size(config):\ndef _display_size(config):", "表示用画像の描画と、入力・表示サイズ設定の検証を行う補助関数群。", [
        "緑: MATCH、橙: MAYBE、赤: NO MATCH。表示は検出計算に影響しない。",
        "frame_width と frame_height は片方だけ設定するとエラーにし、設定ミスを防ぐ。",
    ])
    # types/features
    story += section("3. データ型と特徴量")
    story += subsection("3.1 src/types.py: 結果を受け渡すデータ型")
    story.append(P("このファイルは計算をせず、各段階の結果を決まった形で保持する。処理を分割しても、同じ型を使って結果を渡せるようにしている。"))
    add_function(story, "Template", "@dataclass(frozen=True)\nclass Template:", "倍率別に作った参照テンプレート。BGR画像、GPU特徴量、中心重み、特徴量分散を持つ。", [
        "height / width プロパティは features の形状からテンプレート寸法を返す。", "frozen=True のため、作成後のテンプレート値を意図せず変更しにくい。",
    ], is_class=True)
    add_function(story, "Candidate / Roi / DetailMatch", "@dataclass(frozen=True)\nclass Candidate: ...\nclass Roi: ...\nclass DetailMatch: ...", "粗探索候補、元解像度の詳細探索領域、ROI・倍率ごとの最良候補を表す。", [
        "Candidate の座標は縮小フレーム上、Roi と DetailMatch の座標は元フレーム上。", "DetailMatch は色差score、分散距離、実際に用いたstrideも保持する。",
    ], is_class=True)
    add_function(story, "DetailVarianceFilter / FrameResult", "@dataclass(frozen=True)\nclass DetailVarianceFilter: ...\nclass FrameResult: ...", "詳細探索の分散フィルタ結果と、1フレーム全体の結果を表す。", [
        "candidate_count / passed_count / rejected_count は分散フィルタの候補数を集計するプロパティ。",
        "FrameResult は粗探索結果、ROI、詳細候補、最良候補、最終状態をまとめる。",
    ], is_class=True)
    story += subsection("3.2 src/features.py: BGRから照合用特徴量へ")
    add_function(story, "to_gpu_bgr(image_bgr, device)", "return torch.from_numpy(np.ascontiguousarray(image_bgr)).to(\n    device=device, dtype=torch.float32\n)", "OpenCVのBGR配列を連続メモリのfloat32 GPU Tensorへ転送する。", [
        "元フレームは1フレームにつき1回だけGPUへ送る。", "以降の特徴量計算・探索はGPU上で行う。",
    ])
    add_function(story, "normalise_weights / normalise_channel_weights", "def normalise_weights(values, device):\ndef normalise_channel_weights(values, device):", "Brightness用RGB重みと、RG/BY/Brightness誤差重みをGPU用Tensorにする。", [
        "Brightness用は合計1へ正規化する。", "誤差用は相対強度を保つため、合計1へは正規化しない。",
    ])
    add_function(story, "bgr_to_features(image_bgr, brightness_weights)", "rg = red - green\nby = (red + green) * 0.5 - blue\nbrightness = wr * red + wg * green + wb * blue", "BGRを RG / BY / Brightness の3特徴量へ変換する。", [
        "以降のテンプレート照合・分散比較は、この3値を使う。", "RGは赤-緑、BYは黄系-青系、Brightnessは重み付き明るさ。",
    ])
    add_function(story, "center_weight / downscale_on_gpu / scan_positions", "def center_weight(height, width, min_weight, device):\ndef downscale_on_gpu(image_bgr, scale):\ndef scan_positions(maximum, stride):", "中心重み画像の作成、粗探索用のGPU縮小、走査座標の作成を行う。", [
        "中心重みはテンプレート端部の背景を過度に重視しないために使う。",
        "scan_positions は右端・下端も必ず探索対象に加える。",
    ])
    story.append(PageBreak())

    # pipeline and coarse
    story += section("4. 検出の前半: pipeline.py と coarse_search.py")
    story += subsection("4.1 CoarseToFineMatcher: 検出器の窓口")
    add_function(story, "CoarseToFineMatcher", "class CoarseToFineMatcher:", "粗探索・ROI作成・詳細探索を順番に呼ぶ検出器の窓口。計算の具体的な中身は役割別ファイルへ分けている。", [
        "入力は参照テンプレートと設定値。出力は1フレーム分の FrameResult。", "CUDAが設定されたのに使えない場合は、初期化時に明確なエラーを出す。",
    ], is_class=True)
    add_function(story, "__init__(...)", "self.coarse_templates = self._create_templates(...)\nself.detail_templates = self._create_templates(...)", "設定値を検証し、GPU、重み、粗探索・詳細探索テンプレートを初期化する。", [
        "フレームごとには呼ばれない。", "各倍率の参照特徴量・分散を先に計算し、繰返し計算を減らす。",
    ])
    add_function(story, "detail_stride_for_scale(scale)", "return max(self.detail_stride_min,\n           int(self.detail_stride_base * scale))", "詳細探索時のテンプレート倍率別strideを返す。", [
        "小テンプレートほど細かく探索して、遠距離の小さなマネキンの位置ずれを減らす。",
        "例: base=4, min=2なら、1.0→4、0.9→3、0.7以下→2。",
    ])
    add_function(story, "_create_templates(original_bgr, scales, prefix)", "features = bgr_to_features(tensor, self.brightness_weights)\nfeature_variance = features.var(dim=(0, 1), correction=0)", "参照画像を倍率ごとに縮小し、照合に必要な情報を Template にまとめる。", [
        "同じ画素サイズになる重複倍率は1つにする。", "分散は後段の相対分散フィルタの基準になる。",
    ])
    add_function(story, "process(frame_bgr, ...)", "coarse_candidates = find_coarse_candidates(...)\nrois = make_rois(...)\ndetail_matches, filters = find_detail_matches(...)", "1フレームに対する検出本体。特徴量化→粗探索→ROI→詳細探索をこの順に実行する。", [
        "詳細候補のうちscoreが最小のものをbest_matchに選ぶ。", "best_match.score が final_max_error 以下なら生検出とする。",
    ])
    story += subsection("4.2 粗探索とROI: src/coarse_search.py")
    add_function(story, "find_coarse_candidates(...)", "values, indices = torch.topk(scores.flatten(), k=count, largest=False)\navailable &= distance_squared >= nms_distance_original_px**2", "縮小フレーム全体で粗くテンプレート照合し、ROI候補を選ぶ。", [
        "各粗探索テンプレート倍率から低スコア候補を取り、NMSで近すぎる重複候補を除く。",
        "最終的に coarse_top_k 個だけをCPUへ戻す。",
    ])
    add_function(story, "make_rois(...)", "coarse_uncertainty = coarse_stride / frame_downscale\nroi_width = round(max_template_width + 2 * (...))", "縮小フレーム上の候補を元解像度へ戻し、詳細照合するROIを作る。", [
        "粗探索strideによる位置ずれを余白に含める。", "ROIが元フレームの外に出ないよう切り詰める。",
    ])
    story.append(PageBreak())

    # scoring detail
    story += section("5. 検出の後半: scoring.py と detail_search.py")
    story += subsection("5.1 色差と分散フィルタ: src/scoring.py")
    add_function(story, "ScoreMap", "@dataclass(frozen=True)\nclass ScoreMap:", "走査した全候補の色差スコア、座標、分散フィルタ判定、候補分散をまとめる型。", [
        "粗探索では主にscores、詳細探索では分散関連の全データも使う。",
    ], is_class=True)
    add_function(story, "extract_candidate_patches(...)", "patches = frame_features[\n    y_indices[:, None, :, None],\n    x_indices[None, :, None, :]\n]", "全走査位置の候補パッチをPyTorchの高度インデックスで一括抽出する。", [
        "Pythonの二重ループを避けるため、処理速度に重要な関数。", "抽出結果は「y位置 × x位置 × テンプレート高さ × 幅 × 3特徴量」。",
    ])
    add_function(story, "score_map(...)", "pixel_error = torch.sqrt(torch.sum(\n    difference.square() * channel_weights, dim=3\n))\nunfiltered_scores = torch.sum(pixel_error * template.weights, dim=(1, 2)) / template.weights.sum()", "全候補に対する、中心重み付きの色差をGPU上で計算する。小さいほど参照画像に近い。", [
        "画素ごとにRG/BY/Brightnessの重み付きユークリッド距離を取り、テンプレート内で平均する。",
        "詳細探索では、下記の分散フィルタを通過した候補だけを残す。",
    ])
    story.append(info_box("詳細探索の分散フィルタ", [
        "平坦領域フィルタ: chroma分散とBrightness分散が両方小さい空・壁を除外する。",
        "相対分散フィルタ: 参照テンプレートと候補のRG/BY/Brightness分散が大きく異なる候補を除外する。",
        "除外候補のscoreは無限大に置き換えるため、最良候補には選ばれない。",
    ]))
    story += subsection("5.2 詳細探索: src/detail_search.py")
    add_function(story, "find_detail_matches(...)", "for roi in rois:\n    for template in detail_templates:\n        stride = detail_stride_for_scale(template.scale)\n        candidate_scores = score_map(...)", "各ROIと各詳細テンプレート倍率の組合せを、元解像度で照合する。", [
        "組合せごとに最小スコア1つを DetailMatch として残す。", "全候補が分散フィルタで除外されている場合は DetailMatch を作らない。",
        "可視化やJSON保存が必要なときだけ、全候補の分散判定配列をGPUからCPUへ送る。",
    ])
    # tracking
    story += section("6. 時系列確定: src/tracking.py")
    story.append(P("テンプレート照合で1フレームだけ似ている候補が出ても、すぐにMATCHにはしない。同じ位置に近い生検出が複数フレーム現れた場合だけMATCHにすることで、一時的な誤検出を抑える。"))
    story.append(info_box("状態の意味", [
        "NO MATCH: 色差閾値を通る生検出がない。",
        "MAYBE: 生検出はあるが、直近window内の検出回数がconfirm_frames未満。",
        "MATCH: 同一トラックで、直近window内の生検出回数がconfirm_frames以上。",
    ]))
    add_function(story, "TemporalMatchTracker", "class TemporalMatchTracker:", "IoUで同じ対象を追跡し、履歴からMATCHを確定するクラス。", [
        "window_frames、confirm_frames、match_iou_threshの妥当性を初期化時に確認する。",
    ], is_class=True)
    add_function(story, "_iou_xyxy(first, second)", "intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)\nreturn intersection / max(first_area + second_area - intersection, 1e-6)", "2つの矩形のIoUを返す。", [
        "IoUは重なり面積 ÷ 和集合面積。大きいほど同じ位置・大きさの候補。", "match_iou_thresh以上なら、前フレームまでの同一トラックとして扱う。",
    ])
    add_function(story, "_append_history(track, is_target_frame)", "track.target_history.append(is_target_frame)\ntrack.target_history = track.target_history[-self.window_frames:]", "対象と判断されたかどうかをトラックの履歴へ追加し、指定窓の長さだけ残す。", [
        "Trueの個数がconfirm_frames以上でMATCHになる。",
    ])
    add_function(story, "update(bbox_xyxy)", "if best_iou >= self.match_iou_thresh:\n    self._append_history(updated_track, True)\nconfirmed = positive_frames >= self.confirm_frames", "現在の生検出枠を既存トラックへ対応付け、状態を確定する。", [
        "十分なIoUのトラックがなければ新しいトラックを作る。", "生検出がないフレームではNO MATCHを返す。",
    ])
    # artifacts
    story += section("7. 保存と可視化: src/artifacts.py")
    story.append(P("保存は検出ロジックから切り分けられている。save.enabled を false にすると、画像・CSV・JSON・検証動画を一切作らず、保存用データのCPU転送も必要な場合だけに抑える。"))
    add_function(story, "ArtifactSaveOptions.from_config(config)", "@classmethod\ndef from_config(cls, config):", "YAMLの save: を読み、保存する成果物と間隔を決める。", [
        "旧形式のsave_every_n_frames、save_annotated_videoにも互換対応している。", "has_frame_output / has_any_output は、保存対象があるかを返すプロパティ。",
    ])
    add_function(story, "ArtifactWriter", "class ArtifactWriter:", "実行ごとの結果フォルダを作り、画像・CSV・JSONを出力するクラス。", [
        "run_YYYYMMDD_HHMMSS 形式のフォルダへ結果をまとめる。", "保存無効時は結果フォルダ自体を作らない。",
    ], is_class=True)
    add_function(story, "save_detail_templates(templates)", "template_scale_{template.scale:.3f}.png", "詳細探索用の参照テンプレート画像を 01_detail_templates/ へ保存する。", [
        "どの倍率のテンプレートを使ったか、実験後に確認できる。",
    ])
    add_function(story, "save_frame(frame_number, original_bgr, result)", "for candidate in result.coarse_candidates: ...\nfor variance_filter in result.detail_variance_filters: ...\nfor detail in result.detail_matches: ...", "設定に従い、1フレームの中間画像・数値・最終結果を保存する中心関数。", [
        "元画像、粗探索候補、ROI、分散除外、詳細候補、最良候補、CSV、JSONを選択して出力できる。",
        "JSONには座標・スコア・分散除外数・追跡状態を保存し、後から数値比較できる。",
    ])
    add_function(story, "_draw_box / _draw_variance_rejections", "cv2.rectangle(image, (x, y), (x + width, y + height), colour, 2)", "検証用画像に枠・ラベル・分散除外位置を描画する。", [
        "青: ROI、緑: MATCH、橙: MAYBEまたは相対分散除外、赤: NO MATCH、マゼンタ: 平坦領域除外。",
        "候補矩形全体を描くため、候補が重なるほど色が濃く見える。",
    ])
    add_function(story, "AnnotatedVideoWriter / draw_annotated_frame / _write_image", "self.writer.write(draw_annotated_frame(...))\nencoded.tofile(str(path))", "検証動画の書込み、フレームへのROI・最良候補の描画、日本語パスに対応した画像保存を行う。", [
        "検証動画は 07_annotated_video/coarse_rois_and_best_match.mp4 に保存される。",
        "_write_imageはcv2.imwriteで問題になり得る日本語パスも保存できる。",
    ])
    story.append(PageBreak())

    # performance config usage
    story += section("8. 速度計測・設定・実行")
    story += subsection("8.1 src/performance.py")
    add_function(story, "PerformanceOptions.from_config(config)", "warmup_frames = int(raw.get(\"warmup_frames\", 10))", "performance: 設定から、計測の有効化・ウォームアップ数・途中報告間隔を読む。", [
        "初期GPU処理が遅い影響を避けるため、warmup_frames後だけを集計対象にできる。",
    ])
    add_function(story, "PerformanceTracker", "class PerformanceTracker:\n    def record(self, timing):\n    def summary(self):\n    def format_summary(self):", "フレームごとの入力、検出、保存、表示、合計時間を集計する。", [
        "summaryは平均・中央値・P95・最小/最大時間、平均FPS、最低FPSを返す。", "_summariseは mean_fps=1000/平均ms、min_fps=1000/最遅ms で計算する。",
    ], is_class=True)
    story += subsection("8.2 実行コマンド")
    story.append(signature_box("command", "& .\\env\\Scripts\\python.exe .\\experiments\\coarse_to_fine_mannequin\\run.py `\n    --config .\\experiments\\coarse_to_fine_mannequin\\config\\default.yaml", "PowerShellでの標準実行例。設定を変える場合はまずYAMLをコピーして実験名を付ける。"))
    story += subsection("8.3 主な設定値")
    config_rows = [[P("項目", "small"), P("意味", "small"), P("注意点", "small")]]
    for a, b, c in [
        ("frame_downscale", "粗探索用フレーム倍率", "大きいほど情報は残るが遅くなる"),
        ("coarse_template_scales", "粗探索で使う倍率", "遠距離の対象なら小倍率も必要"),
        ("detail_template_scales", "詳細探索で使う倍率", "誤差の小さい倍率の偏りを確認する"),
        ("detail_stride_base/min", "倍率別の詳細探索間隔", "小倍率ほど小さいstrideが必要"),
        ("final_max_error", "生検出とする色差上限", "小さいほど厳しい"),
        ("variance_log_distance_max", "参照と候補の相対分散差上限", "低すぎると対象も除外する"),
        ("min_chroma/brightness_variance", "平坦領域除外の下限", "白壁・空の誤検出対策"),
        ("target_confirm_* / match_iou_thresh", "時系列MATCHの条件", "反応速度と安定性のトレードオフ"),
    ]:
        config_rows.append([P(f"<font name='{CODE_FONT}'>{a}</font>", "small"), P(b, "small"), P(c, "small")])
    ct = Table(config_rows, colWidths=[56*mm, 56*mm, 62*mm], repeatRows=1)
    ct.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#B8C1C8")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,GRAY]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(ct)
    story.append(PageBreak())

    # Appendix
    story += section("付録. 結果フォルダの見方と引継ぎ時の確認順")
    result_rows = [[P("場所", "small"), P("確認する内容", "small")]]
    for a, b in [
        ("01_detail_templates/", "参照画像が各倍率でどう縮小されたか。"),
        ("02_frames/", "元フレームと粗探索に使った縮小フレーム。"),
        ("03_coarse_match/", "縮小フレーム上の粗探索候補。"),
        ("04_coarse_rois_original/", "元解像度上のROI。対象がROI内に入っているか最初に確認する。"),
        ("05_detail_match/", "詳細照合、分散フィルタ除外、最良候補。背景に吸われる原因を確認する。"),
        ("06_scores/", "CSV・JSONの数値結果。倍率、score、分散通過数、追跡状態を比較する。"),
        ("07_annotated_video/", "ROIと最良候補を重ねた動画。フレーム全体の挙動を確認する。"),
        ("08_performance/", "処理時間・FPSの要約。設定変更前後の速度比較に使う。"),
    ]:
        result_rows.append([P(f"<font name='{CODE_FONT}'>{a}</font>", "small"), P(b, "small")])
    rt = Table(result_rows, colWidths=[61*mm,113*mm], repeatRows=1)
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#B8C1C8")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,GRAY]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(rt)
    story += subsection("引継ぎ時のおすすめ確認順")
    story.extend(bullets([
        "README.mdを読み、目的・実行方法・保存先を把握する。",
        "run.pyのmain()とpipeline.pyのprocess()を読み、処理順を確認する。",
        "04_coarse_rois_original/で、マネキンがROIに含まれているかを見る。",
        "ROIにあるのに検出されない場合は、05_detail_match/で分散フィルタと最小scoreを確認する。",
        "設定値を変えたら、06_scores/と08_performance/を前回結果と比較し、精度と速度を分けて判断する。",
    ]))
    story.append(Spacer(1, 8))
    story.append(info_box("この資料とソースコードの対応", [
        "詳しいコード抜粋と全関数のリファレンスは document/coarse_to_fine_source_reference.md にも保存している。",
        "本PDFは引継ぎ時に全体を掴むための資料、Markdown版は実装を追うための資料として使い分ける。",
    ]))
    doc.build(story)


if __name__ == "__main__":
    build()
    print(OUTPUT)
