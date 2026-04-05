"""Unit tests for whalu.analysis."""

from datetime import date

import polars as pl
import pytest

from whalu.analysis import (
    add_timestamps,
    daily_counts,
    hourly_activity,
    species_summary,
)


def _make_detections(
    species: list[str],
    confidences: list[float],
    ranks: list[int],
    hours: list[int],
    source: str = "mbari/MARS-20260301T000000Z-16kHz",
    dates: list[date] | None = None,
) -> pl.DataFrame:
    """Build a minimal detection DataFrame for testing.

    Parameters
    ----------
    species : list[str]
        Species codes for each row.
    confidences : list[float]
        Confidence values for each row.
    ranks : list[int]
        Rank values for each row.
    hours : list[int]
        Hour-of-day values for each row.
    source : str, optional
        Source string (used as-is for all rows).
    dates : list[date] or None, optional
        Date values. Defaults to 2026-03-01 for all rows.

    Returns
    -------
    pl.DataFrame
        Detection DataFrame with required columns.
    """
    n = len(species)
    if dates is None:
        dates = [date(2026, 3, 1)] * n
    return pl.DataFrame(
        {
            "source": [source] * n,
            "time_start_s": [float(i * 2.5) for i in range(n)],
            "time_end_s": [float(i * 2.5 + 5.0) for i in range(n)],
            "species": species,
            "confidence": confidences,
            "rank": ranks,
            "hour": hours,
            "date": dates,
        }
    )


class TestAddTimestamps:
    """Tests for add_timestamps."""

    def test_adds_timestamp_column(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/MARS-20260301T000000Z-16kHz"],
                "time_start_s": [0.0],
            }
        )
        result = add_timestamps(df)
        assert "timestamp" in result.columns

    def test_adds_hour_and_date_columns(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/MARS-20260301T000000Z-16kHz"],
                "time_start_s": [0.0],
            }
        )
        result = add_timestamps(df)
        assert "hour" in result.columns
        assert "date" in result.columns

    def test_base_timestamp_is_utc(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/MARS-20260301T000000Z-16kHz"],
                "time_start_s": [0.0],
            }
        )
        result = add_timestamps(df)
        ts = result["timestamp"][0]
        assert ts.tzinfo is not None or result["timestamp"].dtype == pl.Datetime(
            "ms", "UTC"
        )

    def test_hour_zero_at_midnight(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/MARS-20260301T000000Z-16kHz"],
                "time_start_s": [0.0],
            }
        )
        result = add_timestamps(df)
        assert result["hour"][0] == 0

    def test_hour_advances_with_time_start(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/MARS-20260301T000000Z-16kHz"],
                "time_start_s": [3600.0],  # 1 hour in
            }
        )
        result = add_timestamps(df)
        assert result["hour"][0] == 1

    def test_date_correct(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/MARS-20260301T000000Z-16kHz"],
                "time_start_s": [0.0],
            }
        )
        result = add_timestamps(df)
        assert result["date"][0] == date(2026, 3, 1)

    def test_invalid_source_uses_fallback_date(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/unknown-file.wav"],
                "time_start_s": [0.0],
            }
        )
        result = add_timestamps(df)
        assert result["date"][0] == date(2000, 1, 1)

    def test_row_count_preserved(self):
        df = pl.DataFrame(
            {
                "source": ["mbari/MARS-20260301T000000Z-16kHz"] * 10,
                "time_start_s": [float(i * 2.5) for i in range(10)],
            }
        )
        result = add_timestamps(df)
        assert len(result) == 10

    def test_multiple_sources(self):
        df = pl.DataFrame(
            {
                "source": [
                    "mbari/MARS-20260301T000000Z-16kHz",
                    "mbari/MARS-20260302T120000Z-16kHz",
                ],
                "time_start_s": [0.0, 0.0],
            }
        )
        result = add_timestamps(df)
        assert result["date"][0] == date(2026, 3, 1)
        assert result["date"][1] == date(2026, 3, 2)
        assert result["hour"][1] == 12


class TestSpeciesSummary:
    """Tests for species_summary."""

    def test_returns_dataframe(self):
        df = _make_detections(
            species=["Bm", "Bm", "Bp"],
            confidences=[0.8, 0.9, 0.7],
            ranks=[1, 1, 1],
            hours=[0, 1, 2],
        )
        result = species_summary(df)
        assert isinstance(result, pl.DataFrame)

    def test_filters_rank_gt_1(self):
        df = _make_detections(
            species=["Bm", "Bm"],
            confidences=[0.8, 0.8],
            ranks=[1, 2],
            hours=[0, 0],
        )
        result = species_summary(df)
        assert len(result) == 1
        assert result["windows"][0] == 1

    def test_filters_low_confidence(self):
        df = _make_detections(
            species=["Bm", "Bm"],
            confidences=[0.8, 0.3],
            ranks=[1, 1],
            hours=[0, 1],
        )
        result = species_summary(df)
        assert result["windows"][0] == 1

    def test_expected_columns(self):
        df = _make_detections(
            species=["Bm"],
            confidences=[0.9],
            ranks=[1],
            hours=[0],
        )
        result = species_summary(df)
        for col in [
            "species",
            "windows",
            "minutes",
            "mean_conf",
            "max_conf",
            "pct_of_time",
        ]:
            assert col in result.columns

    def test_minutes_computed_from_windows(self):
        df = _make_detections(
            species=["Bm"] * 12,
            confidences=[0.9] * 12,
            ranks=[1] * 12,
            hours=list(range(12)),
        )
        result = species_summary(df)
        # 12 windows * 2.5s / 60 = 0.5 minutes
        assert pytest.approx(result["minutes"][0], abs=0.01) == 0.5

    def test_sorted_by_windows_descending(self):
        df = _make_detections(
            species=["Bp", "Bm", "Bm", "Bm"],
            confidences=[0.9, 0.9, 0.9, 0.9],
            ranks=[1, 1, 1, 1],
            hours=[0, 1, 2, 3],
        )
        result = species_summary(df)
        assert result["species"][0] == "Bm"

    def test_empty_after_filtering(self):
        df = _make_detections(
            species=["Bm"],
            confidences=[0.1],
            ranks=[1],
            hours=[0],
        )
        result = species_summary(df)
        assert len(result) == 0


class TestHourlyActivity:
    """Tests for hourly_activity."""

    def test_returns_24_rows(self):
        df = _make_detections(
            species=["Bm"] * 5,
            confidences=[0.9] * 5,
            ranks=[1] * 5,
            hours=[0, 1, 2, 3, 4],
        )
        hourly, _ = hourly_activity(df)
        assert len(hourly) == 24

    def test_hour_column_is_0_to_23(self):
        df = _make_detections(
            species=["Bm"],
            confidences=[0.9],
            ranks=[1],
            hours=[0],
        )
        hourly, _ = hourly_activity(df)
        assert hourly["hour"].to_list() == list(range(24))

    def test_missing_hours_filled_with_zero(self):
        # Only hour 5 has data
        df = _make_detections(
            species=["Bm"],
            confidences=[0.9],
            ranks=[1],
            hours=[5],
        )
        hourly, _ = hourly_activity(df)
        # All other hours should be 0
        other_hours = hourly.filter(pl.col("hour") != 5)
        assert (other_hours["Bm"] == 0.0).all()

    def test_top_n_respected(self):
        species_list = ["Bm", "Bp", "Mn", "Ba", "Bs", "Be"]
        df = _make_detections(
            species=species_list,
            confidences=[0.9] * 6,
            ranks=[1] * 6,
            hours=[0] * 6,
        )
        _, top = hourly_activity(df, top_n=3)
        assert len(top) == 3

    def test_top_species_ordered_by_count(self):
        # Bm appears 3 times, Bp appears 1 time
        df = _make_detections(
            species=["Bm", "Bm", "Bm", "Bp"],
            confidences=[0.9, 0.9, 0.9, 0.9],
            ranks=[1, 1, 1, 1],
            hours=[0, 1, 2, 3],
        )
        _, top = hourly_activity(df, top_n=2)
        assert top[0] == "Bm"

    def test_filters_low_confidence(self):
        df = _make_detections(
            species=["Bm", "Bm"],
            confidences=[0.9, 0.1],
            ranks=[1, 1],
            hours=[0, 1],
        )
        hourly, top = hourly_activity(df)
        assert "Bm" in top
        # Only 1 detection above threshold
        assert hourly.filter(pl.col("hour") == 0)["Bm"][0] > 0
        assert hourly.filter(pl.col("hour") == 1)["Bm"][0] == 0.0

    def test_rate_column_exists_for_top_species(self):
        df = _make_detections(
            species=["Bm"],
            confidences=[0.9],
            ranks=[1],
            hours=[0],
        )
        hourly, top = hourly_activity(df, top_n=1)
        assert "Bm" in hourly.columns


class TestDailyCounts:
    """Tests for daily_counts."""

    def test_returns_dataframe(self):
        df = _make_detections(
            species=["Bm"],
            confidences=[0.9],
            ranks=[1],
            hours=[0],
        )
        result = daily_counts(df)
        assert isinstance(result, pl.DataFrame)

    def test_expected_columns(self):
        df = _make_detections(
            species=["Bm"],
            confidences=[0.9],
            ranks=[1],
            hours=[0],
        )
        result = daily_counts(df)
        for col in ["date", "species", "windows"]:
            assert col in result.columns

    def test_filters_rank_gt_1(self):
        df = _make_detections(
            species=["Bm", "Bm"],
            confidences=[0.9, 0.9],
            ranks=[1, 2],
            hours=[0, 0],
        )
        result = daily_counts(df)
        assert result["windows"].sum() == 1

    def test_filters_low_confidence(self):
        df = _make_detections(
            species=["Bm", "Bm"],
            confidences=[0.9, 0.3],
            ranks=[1, 1],
            hours=[0, 1],
        )
        result = daily_counts(df)
        assert result["windows"].sum() == 1

    def test_groups_by_date_and_species(self):
        d1 = date(2026, 3, 1)
        d2 = date(2026, 3, 2)
        df = _make_detections(
            species=["Bm", "Bm", "Bp"],
            confidences=[0.9, 0.9, 0.9],
            ranks=[1, 1, 1],
            hours=[0, 12, 6],
            dates=[d1, d2, d1],
        )
        result = daily_counts(df)
        assert len(result) == 3  # Bm/d1, Bm/d2, Bp/d1

    def test_sorted_by_date_and_species(self):
        d1 = date(2026, 3, 1)
        df = _make_detections(
            species=["Bp", "Bm"],
            confidences=[0.9, 0.9],
            ranks=[1, 1],
            hours=[0, 0],
            dates=[d1, d1],
        )
        result = daily_counts(df)
        # Sorted by date then species
        assert result["species"][0] == "Bm"
        assert result["species"][1] == "Bp"
