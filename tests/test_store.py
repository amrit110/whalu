"""Unit tests for whalu.db.store."""

from pathlib import Path

import polars as pl
import pytest

from whalu.db.store import DetectionStore


def _sample_df(
    n: int = 3, species: str = "Bm", rank: int = 1, confidence: float = 0.9
) -> pl.DataFrame:
    """Create a minimal detection DataFrame.

    Parameters
    ----------
    n : int
        Number of rows.
    species : str
        Species code for all rows.
    rank : int
        Rank value for all rows.
    confidence : float
        Confidence value for all rows.

    Returns
    -------
    pl.DataFrame
        Detection DataFrame.
    """
    return pl.DataFrame(
        {
            "source": ["test"] * n,
            "time_start_s": [float(i * 2.5) for i in range(n)],
            "time_end_s": [float(i * 2.5 + 5.0) for i in range(n)],
            "species": [species] * n,
            "confidence": [confidence] * n,
            "rank": [rank] * n,
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> DetectionStore:
    """Provide a DetectionStore backed by a temporary directory."""
    return DetectionStore(tmp_path / "detections")


class TestDetectionStoreInit:
    """Tests for DetectionStore.__init__."""

    def test_creates_output_dir(self, tmp_path: Path):
        output = tmp_path / "new_dir" / "nested"
        DetectionStore(output)
        assert output.exists()

    def test_accepts_string_path(self, tmp_path: Path):
        store = DetectionStore(str(tmp_path / "detections"))
        assert store.output_dir.exists()

    def test_output_dir_is_path(self, store: DetectionStore):
        assert isinstance(store.output_dir, Path)


class TestIsDone:
    """Tests for DetectionStore.is_done."""

    def test_returns_false_for_new_stem(self, store: DetectionStore):
        assert not store.is_done("test_file")

    def test_returns_true_after_write(self, store: DetectionStore):
        store.write(_sample_df(), "test_file")
        assert store.is_done("test_file")

    def test_different_stems_independent(self, store: DetectionStore):
        store.write(_sample_df(), "file_a")
        assert store.is_done("file_a")
        assert not store.is_done("file_b")


class TestWrite:
    """Tests for DetectionStore.write."""

    def test_returns_path(self, store: DetectionStore):
        p = store.write(_sample_df(), "test_file")
        assert isinstance(p, Path)

    def test_file_exists_after_write(self, store: DetectionStore):
        p = store.write(_sample_df(), "test_file")
        assert p.exists()

    def test_file_is_parquet(self, store: DetectionStore):
        p = store.write(_sample_df(), "test_file")
        assert p.suffix == ".parquet"

    def test_stem_in_filename(self, store: DetectionStore):
        p = store.write(_sample_df(), "my_audio_file")
        assert "my_audio_file" in p.name

    def test_overwrites_existing(self, store: DetectionStore):
        df1 = _sample_df(n=2)
        df2 = _sample_df(n=5)
        store.write(df1, "test_file")
        store.write(df2, "test_file")
        result = pl.read_parquet(store._path("test_file"))
        assert len(result) == 5


class TestMerge:
    """Tests for DetectionStore.merge."""

    def test_empty_store_returns_empty_dataframe(self, store: DetectionStore):
        result = store.merge()
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_single_file_returns_its_contents(self, store: DetectionStore):
        df = _sample_df(n=4)
        store.write(df, "file_a")
        result = store.merge()
        assert len(result) == 4

    def test_multiple_files_concatenated(self, store: DetectionStore):
        store.write(_sample_df(n=3), "file_a")
        store.write(_sample_df(n=5), "file_b")
        result = store.merge()
        assert len(result) == 8

    def test_merge_preserves_schema(self, store: DetectionStore):
        df = _sample_df()
        store.write(df, "file_a")
        result = store.merge()
        for col in df.columns:
            assert col in result.columns


class TestSummary:
    """Tests for DetectionStore.summary."""

    def test_empty_store_returns_empty_dataframe(self, store: DetectionStore):
        result = store.summary()
        assert result.is_empty()

    def test_expected_columns(self, store: DetectionStore):
        store.write(_sample_df(), "file_a")
        result = store.summary()
        for col in ["species", "windows", "minutes_detected", "max_conf", "mean_conf"]:
            assert col in result.columns

    def test_filters_rank_gt_1(self, store: DetectionStore):
        store.write(_sample_df(rank=2), "file_a")
        result = store.summary()
        assert result.is_empty()

    def test_filters_low_confidence(self, store: DetectionStore):
        store.write(_sample_df(confidence=0.04), "file_a")
        result = store.summary()
        assert result.is_empty()

    def test_counts_windows(self, store: DetectionStore):
        store.write(_sample_df(n=10), "file_a")
        result = store.summary()
        assert result["windows"][0] == 10

    def test_minutes_detected(self, store: DetectionStore):
        store.write(_sample_df(n=6), "file_a")
        result = store.summary()
        # 6 windows * 2.5s / 60 = 0.25 minutes
        assert pytest.approx(result["minutes_detected"][0], abs=0.001) == 0.25

    def test_sorted_by_windows_descending(self, store: DetectionStore):
        df = pl.concat([_sample_df(n=5, species="Bm"), _sample_df(n=2, species="Bp")])
        store.write(df, "file_a")
        result = store.summary()
        assert result["species"][0] == "Bm"

    def test_hop_size_parameter(self, store: DetectionStore):
        store.write(_sample_df(n=4), "file_a")
        result = store.summary(hop_size_s=5.0)
        # 4 windows * 5.0s / 60
        assert pytest.approx(result["minutes_detected"][0], abs=0.001) == 4 * 5.0 / 60
