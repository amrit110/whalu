"""Unit tests for whalu.detection.runner."""

from dataclasses import dataclass

import numpy as np
import pytest

from whalu.detection.runner import HOP_SIZE_S, run_detections


@dataclass
class _ClassList:
    classes: list[str]


@dataclass
class _Outputs:
    logits: dict[str, np.ndarray]


class _MockModel:
    """Minimal mock of a Perch embedding model.

    Parameters
    ----------
    classes : list[str]
        Class labels returned by the model.
    logits : np.ndarray or None
        Fixed logits to return for every window. If None, defaults to all-zeros.
    sample_rate : int
        Model sample rate in Hz.
    window_size_s : float
        Window duration in seconds.
    logits_key : str
        Key used to look up logits in the output dict.
    """

    def __init__(
        self,
        classes: list[str],
        logits: np.ndarray | None = None,
        sample_rate: int = 16_000,
        window_size_s: float = 5.0,
        logits_key: str = "multispecies_whale",
    ):
        self.sample_rate = sample_rate
        self.window_size_s = window_size_s
        self.class_list = _ClassList(classes=classes)
        self._logits = logits if logits is not None else np.zeros(len(classes))
        self._logits_key = logits_key

    def embed(self, audio: np.ndarray) -> _Outputs:
        return _Outputs(logits={self._logits_key: self._logits})


CLASSES = ["Bm", "Bp", "Mn", "Ba", "Bs"]
SR = 16_000
WINDOW_S = 5.0
WINDOW_SAMPLES = int(WINDOW_S * SR)
# 3 full windows: 5s window, 2.5s hop → 3 windows from 10s audio
AUDIO_10S = np.zeros(int(10.0 * SR), dtype=np.float32)


class TestRunDetectionsSchema:
    """Tests for output schema of run_detections."""

    def test_returns_dataframe(self):
        import polars as pl

        model = _MockModel(CLASSES)
        result = run_detections(model, AUDIO_10S, "test_source")
        assert isinstance(result, pl.DataFrame)

    def test_expected_columns(self):
        model = _MockModel(CLASSES)
        result = run_detections(model, AUDIO_10S, "test_source")
        for col in [
            "source",
            "time_start_s",
            "time_end_s",
            "species",
            "confidence",
            "rank",
        ]:
            assert col in result.columns

    def test_source_name_in_output(self):
        logits = np.array([2.0, -1.0, -1.0, -1.0, -1.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "my_source")
        assert (result["source"] == "my_source").all()


class TestRunDetectionsFiltering:
    """Tests for detection thresholding in run_detections."""

    def test_no_detections_when_all_logits_negative(self):
        # All logits < 0 → sigmoid < 0.5 → no detections
        logits = np.full(len(CLASSES), -2.0)
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        assert len(result) == 0

    def test_detections_when_logits_positive(self):
        # logit > 0 → sigmoid > 0.5 → detected
        logits = np.array([2.0, -2.0, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        assert len(result) > 0

    def test_only_above_threshold_species_emitted(self):
        # Only Bm (index 0) above threshold
        logits = np.array([2.0, -2.0, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        assert set(result["species"].to_list()) == {"Bm"}

    def test_custom_threshold(self):
        # sigmoid(1.0) ≈ 0.73, use threshold 0.8 to exclude it
        logits = np.array([1.0, -2.0, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test", threshold=0.8)
        assert len(result) == 0

    def test_all_species_above_threshold(self):
        logits = np.full(len(CLASSES), 3.0)
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        # Each window emits all 5 species
        unique_species = set(result["species"].to_list())
        assert unique_species == set(CLASSES)


class TestRunDetectionsRanking:
    """Tests for per-window rank ordering in run_detections."""

    def test_rank_starts_at_1(self):
        logits = np.array([2.0, 1.5, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        assert result["rank"].min() == 1

    def test_rank_1_has_highest_confidence(self):
        # Bm logit 3.0 > Bp logit 1.5, so Bm should be rank 1
        logits = np.array([3.0, 1.5, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        rank1 = result.filter(result["rank"] == 1)
        # All rank-1 rows should be Bm (highest logit)
        assert (rank1["species"] == "Bm").all()

    def test_rank_monotone_per_window(self):
        logits = np.array([3.0, 2.0, 1.5, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        # For each unique (source, time_start_s), ranks should be 1,2,3,...
        for (start,), group in result.group_by("time_start_s"):
            ranks = sorted(group["rank"].to_list())
            assert ranks == list(range(1, len(ranks) + 1))


class TestRunDetectionsWindowing:
    """Tests for sliding-window behaviour of run_detections."""

    def test_audio_shorter_than_window_produces_no_output(self):
        short_audio = np.zeros(WINDOW_SAMPLES // 2, dtype=np.float32)
        logits = np.full(len(CLASSES), 3.0)
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, short_audio, "test")
        assert len(result) == 0

    def test_time_start_increases_by_hop(self):
        logits = np.array([2.0, -2.0, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        starts = sorted(result["time_start_s"].unique().to_list())
        for i in range(1, len(starts)):
            assert pytest.approx(starts[i] - starts[i - 1], abs=0.01) == HOP_SIZE_S

    def test_time_end_equals_start_plus_window(self):
        logits = np.array([2.0, -2.0, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result = run_detections(model, AUDIO_10S, "test")
        diffs = (result["time_end_s"] - result["time_start_s"]).to_list()
        for d in diffs:
            assert pytest.approx(d, abs=0.01) == WINDOW_S

    def test_offset_shifts_timestamps(self):
        logits = np.array([2.0, -2.0, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits)
        result_no_offset = run_detections(model, AUDIO_10S, "test", offset_s=0.0)
        result_offset = run_detections(model, AUDIO_10S, "test", offset_s=3600.0)
        starts_no_offset = sorted(result_no_offset["time_start_s"].unique().to_list())
        starts_offset = sorted(result_offset["time_start_s"].unique().to_list())
        for a, b in zip(starts_no_offset, starts_offset):
            assert pytest.approx(b - a, abs=0.01) == 3600.0


class TestRunDetectionsLogitHandling:
    """Tests for multi-dimensional logit squeezing."""

    def test_2d_logits_are_squeezed(self):
        # Shape (1, n_classes) should be squeezed to (n_classes,)
        logits_2d = np.array([[2.0, -2.0, -2.0, -2.0, -2.0]])
        model = _MockModel(CLASSES, logits=logits_2d)
        result = run_detections(model, AUDIO_10S, "test")
        assert len(result) > 0
        assert (result["species"] == "Bm").all()

    def test_3d_logits_are_squeezed(self):
        logits_3d = np.array([[[2.0, -2.0, -2.0, -2.0, -2.0]]])
        model = _MockModel(CLASSES, logits=logits_3d)
        result = run_detections(model, AUDIO_10S, "test")
        assert len(result) > 0

    def test_custom_logits_key(self):
        logits = np.array([2.0, -2.0, -2.0, -2.0, -2.0])
        model = _MockModel(CLASSES, logits=logits, logits_key="surfperch")
        result = run_detections(model, AUDIO_10S, "test", logits_key="surfperch")
        assert len(result) > 0
