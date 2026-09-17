from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.coarse_to_fine_mannequin.run import apply_temporal_tracking
from experiments.coarse_to_fine_mannequin.src.pipeline import FrameResult
from experiments.coarse_to_fine_mannequin.src.tracking import TemporalMatchTracker


class TemporalMatchTrackerTests(unittest.TestCase):
    def test_confirms_after_five_nearby_raw_matches(self) -> None:
        tracker = TemporalMatchTracker(window_frames=10, confirm_frames=5, match_iou_thresh=0.8)

        decisions = [tracker.update((48, 60, 60, 76)) for _ in range(5)]

        self.assertEqual([item.status for item in decisions[:4]], ["MAYBE"] * 4)
        self.assertTrue(decisions[4].confirmed)
        self.assertEqual(decisions[4].positive_frames, 5)
        self.assertEqual(decisions[4].track_id, 0)

    def test_distant_detection_starts_another_maybe_track(self) -> None:
        tracker = TemporalMatchTracker(window_frames=10, confirm_frames=5, match_iou_thresh=0.8)
        for _ in range(4):
            tracker.update((48, 60, 60, 76))

        decision = tracker.update((4, 4, 16, 20))

        self.assertEqual(decision.status, "MAYBE")
        self.assertEqual(decision.track_id, 1)
        self.assertEqual(tracker.track_info[0].target_history, [True] * 4 + [False])
        self.assertEqual(tracker.track_info[1].target_history, [True])

    def test_no_raw_match_returns_no_match_and_adds_a_negative_history(self) -> None:
        tracker = TemporalMatchTracker(window_frames=10, confirm_frames=5)
        tracker.update((48, 60, 60, 76))

        decision = tracker.update(None)

        self.assertEqual(decision.status, "NO MATCH")
        self.assertEqual(tracker.track_info[0].target_history, [True, False])

    def test_run_result_only_becomes_detected_after_confirmation(self) -> None:
        tracker = TemporalMatchTracker(window_frames=10, confirm_frames=5, match_iou_thresh=0.8)
        best_match = SimpleNamespace(
            x=48, y=60, template=SimpleNamespace(width=12, height=16)
        )
        raw_result = FrameResult(
            coarse_image_bgr=None,
            coarse_candidates=[],
            rois=[],
            detail_matches=[],
            detail_variance_filters=[],
            best_match=best_match,
            detected=True,
        )

        results = [apply_temporal_tracking(raw_result, tracker) for _ in range(5)]

        self.assertEqual([result.temporal_status for result in results[:4]], ["MAYBE"] * 4)
        self.assertTrue(results[4].detected)
        self.assertEqual(results[4].temporal_status, "MATCH")
        self.assertEqual(results[4].positive_frames, 5)


if __name__ == "__main__":
    unittest.main()
