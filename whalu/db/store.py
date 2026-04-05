"""Parquet-based detection store with per-file checkpointing."""

from pathlib import Path

import polars as pl


class DetectionStore:
    """
    Writes one Parquet file per audio file processed.
    Skips files already on disk → safe to resume interrupted runs.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stem: str) -> Path:
        return self.output_dir / f"{stem}.parquet"

    def is_done(self, stem: str) -> bool:
        """Return True if a Parquet file for the given stem already exists.

        Parameters
        ----------
        stem : str
            Audio file stem (no extension) used as the Parquet filename.

        Returns
        -------
        bool
            True if the output file exists, False otherwise.
        """
        return self._path(stem).exists()

    def write(self, df: pl.DataFrame, stem: str) -> Path:
        """Write a detection DataFrame to ``<output_dir>/<stem>.parquet``.

        Parameters
        ----------
        df : pl.DataFrame
            Detection rows to persist.
        stem : str
            Audio file stem used as the base filename.

        Returns
        -------
        Path
            Absolute path of the written Parquet file.
        """
        p = self._path(stem)
        df.write_parquet(p)
        return p

    def merge(self) -> pl.DataFrame:
        """Concatenate all stored Parquet files into one DataFrame.

        Returns
        -------
        pl.DataFrame
            Combined DataFrame, or an empty DataFrame if no files exist.
        """
        files = sorted(self.output_dir.glob("*.parquet"))
        if not files:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(f) for f in files])

    def summary(self, hop_size_s: float = 2.5) -> pl.DataFrame:
        """Compute a top-species summary across all stored detections.

        Only rank-1 detections with confidence > 0.05 are counted.
        ``minutes_detected`` is windows x hop_size_s / 60, representing
        unique non-overlapping time with that species dominant.

        Parameters
        ----------
        hop_size_s : float, optional
            Hop size used during inference (seconds). Default is 2.5.

        Returns
        -------
        pl.DataFrame
            One row per species with columns ``species``, ``windows``,
            ``minutes_detected``, ``max_conf``, and ``mean_conf``, sorted
            by ``windows`` descending. Empty if no data has been stored.
        """
        df = self.merge()
        if df.is_empty():
            return df
        return (
            df.filter((pl.col("rank") == 1) & (pl.col("confidence") > 0.05))
            .group_by("species")
            .agg(
                pl.len().alias("windows"),
                (pl.len() * hop_size_s / 60).alias("minutes_detected"),
                pl.col("confidence").max().alias("max_conf"),
                pl.col("confidence").mean().alias("mean_conf"),
            )
            .sort("windows", descending=True)
        )
