"""Detection analysis: temporal patterns, species distributions."""

import re
from datetime import datetime, timezone

import polars as pl


def add_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """Parse absolute UTC timestamps from the source column and time_start_s.

    Source format: ``mbari/MARS-20260301T000000Z-16kHz``

    Parameters
    ----------
    df : pl.DataFrame
        Detection DataFrame with ``source`` (str) and ``time_start_s`` (float)
        columns.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with three additional columns: ``timestamp``
        (Datetime[ms, UTC]), ``hour`` (int, 0-23), and ``date`` (Date).
    """
    # Extract YYYYMMDD and HHMMSS from source string
    dates = []
    for source in df["source"].to_list():
        m = re.search(r"(\d{8})T(\d{6})", source)
        if m:
            base = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        else:
            base = datetime(2000, 1, 1, tzinfo=timezone.utc)
        dates.append(base)

    base_ts = [int(d.timestamp()) for d in dates]

    return df.with_columns(
        (
            (pl.Series("_base_ts", base_ts) + pl.col("time_start_s").cast(pl.Int64))
            * 1_000
        )
        .cast(pl.Datetime("ms", "UTC"))
        .alias("timestamp")
    ).with_columns(
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.date().alias("date"),
    )


def species_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-species totals across all detections.

    Considers only rank-1 detections with confidence >= 0.5.

    Parameters
    ----------
    df : pl.DataFrame
        Detection DataFrame with ``rank``, ``confidence``, ``species``,
        ``source``, and ``time_end_s`` columns.

    Returns
    -------
    pl.DataFrame
        One row per species with columns ``species``, ``windows``,
        ``minutes``, ``mean_conf``, ``max_conf``, and ``pct_of_time``,
        sorted by ``windows`` descending.
    """
    rank1 = df.filter((pl.col("rank") == 1) & (pl.col("confidence") >= 0.5))
    # Estimate total windows from time span per source (non-detection windows
    # produce no rows, so we can't count them directly from the dataframe)
    total_windows = int(
        df.group_by("source")
        .agg((pl.col("time_end_s").max() / 2.5).alias("n"))["n"]
        .sum()
    )
    return (
        rank1.group_by("species")
        .agg(
            pl.len().alias("windows"),
            (pl.len() * 2.5 / 60).alias("minutes"),
            pl.col("confidence").mean().alias("mean_conf"),
            pl.col("confidence").max().alias("max_conf"),
        )
        .with_columns(
            (pl.col("windows") / total_windows * 100).round(1).alias("pct_of_time")
        )
        .sort("windows", descending=True)
    )


def hourly_activity(df: pl.DataFrame, top_n: int = 5) -> tuple[pl.DataFrame, list[str]]:
    """Compute detection rate (%) per species per hour of day.

    Uses a fixed denominator of 1440 windows per hour (3600 s / 2.5 s hop).
    Missing hours are filled with 0.

    Parameters
    ----------
    df : pl.DataFrame
        Detection DataFrame with ``rank``, ``confidence``, ``species``,
        and ``hour`` columns.
    top_n : int, optional
        Number of most-detected species to include. Default is 5.

    Returns
    -------
    pl.DataFrame
        Wide DataFrame with 24 rows (one per hour) and one column per top
        species containing the detection rate as a percentage.
    list[str]
        Species codes of the top-N species, ordered by total detections.
    """
    rank1 = df.filter((pl.col("rank") == 1) & (pl.col("confidence") >= 0.5))

    top_species = (
        rank1.group_by("species")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)["species"]
        .to_list()
    )

    # Fixed denominator: 3600s / 2.5s hop = 1440 windows per hour
    WINDOWS_PER_HOUR = 3600 / 2.5

    # Detections per species per hour
    counts = (
        rank1.filter(pl.col("species").is_in(top_species))
        .group_by(["hour", "species"])
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / WINDOWS_PER_HOUR * 100).round(1).alias("rate"))
        .pivot(on="species", index="hour", values="rate", aggregate_function="mean")
        .sort("hour")
    )

    # Fill missing hours/species with 0
    all_hours = pl.DataFrame({"hour": list(range(24))})
    counts = all_hours.join(counts, on="hour", how="left").fill_null(0.0)

    return counts, top_species


def daily_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Count rank-1 detections per day per species.

    Parameters
    ----------
    df : pl.DataFrame
        Detection DataFrame with ``rank``, ``confidence``, ``date``, and
        ``species`` columns.

    Returns
    -------
    pl.DataFrame
        One row per (date, species) pair with column ``windows``, sorted
        by date then species.
    """
    return (
        df.filter((pl.col("rank") == 1) & (pl.col("confidence") >= 0.5))
        .group_by(["date", "species"])
        .agg(pl.len().alias("windows"))
        .sort(["date", "species"])
    )
