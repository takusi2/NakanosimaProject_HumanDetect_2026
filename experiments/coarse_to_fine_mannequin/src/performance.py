"""粗探索・詳細照合実験の処理時間をフレーム単位で集計する。"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Mapping


@dataclass(frozen=True)
class PerformanceOptions:
    """計測表示の設定。"""

    enabled: bool = True
    warmup_frames: int = 10
    report_every_n_frames: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "PerformanceOptions":
        raw = config.get("performance", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("performance must be a YAML mapping")
        warmup_frames = int(raw.get("warmup_frames", 10))
        report_every_n_frames = int(raw.get("report_every_n_frames", 0))
        if warmup_frames < 0 or report_every_n_frames < 0:
            raise ValueError("performance frame counts must be non-negative")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            warmup_frames=warmup_frames,
            report_every_n_frames=report_every_n_frames,
        )


@dataclass(frozen=True)
class FrameTiming:
    """1フレームで計測した処理時間。単位はms。"""

    frame_number: int
    decode_ms: float
    detection_ms: float
    artifact_write_ms: float
    display_ms: float
    total_ms: float


class PerformanceTracker:
    """ウォームアップ後の時間を集計し、端末・JSON向けの要約を作る。"""

    def __init__(self, options: PerformanceOptions) -> None:
        self.options = options
        self._timings: list[FrameTiming] = []

    def record(self, timing: FrameTiming) -> None:
        if timing.frame_number > self.options.warmup_frames:
            self._timings.append(timing)

    def should_report(self, frame_number: int) -> bool:
        return (
            self.options.report_every_n_frames > 0
            and frame_number > self.options.warmup_frames
            and frame_number % self.options.report_every_n_frames == 0
        )

    def summary(self) -> dict[str, object]:
        if not self._timings:
            return {
                "frames_measured": 0,
                "warmup_frames": self.options.warmup_frames,
                "message": "No frames remained after warmup.",
            }
        return {
            "frames_measured": len(self._timings),
            "warmup_frames": self.options.warmup_frames,
            "decode": _summarise([item.decode_ms for item in self._timings]),
            "detection": _summarise([item.detection_ms for item in self._timings]),
            "artifact_write": _summarise(
                [item.artifact_write_ms for item in self._timings]
            ),
            "display": _summarise([item.display_ms for item in self._timings]),
            "total": _summarise([item.total_ms for item in self._timings]),
        }

    def format_summary(self) -> str:
        summary = self.summary()
        if summary["frames_measured"] == 0:
            return "[performance] no measured frames (increase max_frames or reduce warmup_frames)"
        total = summary["total"]
        detection = summary["detection"]
        decode = summary["decode"]
        artifact_write = summary["artifact_write"]
        display = summary["display"]
        return "\n".join(
            (
                f"[performance] measured={summary['frames_measured']} "
                f"warmup={summary['warmup_frames']}",
                _format_stage("total", total),
                _format_stage("detection", detection),
                _format_stage("decode", decode),
                _format_stage("artifact_write", artifact_write),
                _format_stage("display", display),
            )
        )


def _summarise(milliseconds: list[float]) -> dict[str, float]:
    mean_ms = statistics.fmean(milliseconds)
    sorted_values = sorted(milliseconds)
    percentile_index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * 0.95))
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(milliseconds),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "p95_ms": sorted_values[percentile_index],
        "mean_fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        "min_fps": 1000.0 / max(milliseconds) if max(milliseconds) > 0 else float("inf"),
    }


def _format_stage(name: str, values: object) -> str:
    assert isinstance(values, dict)
    return (
        f"  {name}: mean={values['mean_ms']:.2f}ms "
        f"median={values['median_ms']:.2f}ms p95={values['p95_ms']:.2f}ms "
        f"fps(mean)={values['mean_fps']:.2f} fps(min)={values['min_fps']:.2f}"
    )
