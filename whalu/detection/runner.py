"""Sliding-window inference over audio.

The multispecies_whale (and surfperch) models are multi-label binary classifiers —
each class has an independent sigmoid output, NOT a shared softmax.
We apply sigmoid to raw logits and only emit detections above a threshold.
"""

import logging

import numpy as np
import polars as pl
from scipy.special import expit  # sigmoid

log = logging.getLogger(__name__)

HOP_SIZE_S = 2.5  # seconds between window starts
THRESHOLD = (
    0.5  # sigmoid threshold — logit > 0 means the model is positive on this class
)


def run_detections(
    model,
    audio: np.ndarray,
    source_name: str,
    logits_key: str = "multispecies_whale",
    threshold: float = THRESHOLD,
    offset_s: float = 0.0,
) -> pl.DataFrame:
    """Slide a window over audio, run model inference, and return detections.

    Uses sigmoid activation, which is correct for multi-label classifiers
    (independent per-class outputs, not softmax). Only emits rows where
    confidence >= threshold. Windows with no species above threshold
    produce no rows.

    Parameters
    ----------
    model : object
        Embedding model with ``sample_rate`` (int), optional
        ``window_size_s`` (float, default 5.0), ``class_list.classes``
        (list[str]), and ``embed(audio_chunk)`` returning an object with
        ``.logits[logits_key]`` as a numpy array.
    audio : np.ndarray
        Float32 audio array at ``model.sample_rate``.
    source_name : str
        Identifier written into the ``source`` column of every output row.
    logits_key : str, optional
        Key used to retrieve logits from the model output dict.
        Default is ``"multispecies_whale"``.
    threshold : float, optional
        Minimum sigmoid confidence to include a detection. Default is 0.5.
    offset_s : float, optional
        Time offset (seconds) added to all ``time_start_s`` / ``time_end_s``
        values. Useful when processing a file in chunks. Default is 0.0.

    Returns
    -------
    pl.DataFrame
        Detection rows with schema: ``source`` (str), ``time_start_s``
        (float), ``time_end_s`` (float), ``species`` (str),
        ``confidence`` (float), ``rank`` (int). Rank 1 is the
        highest-confidence species in each window.
    """
    sr: int = model.sample_rate
    window_size_s: float = getattr(model, "window_size_s", 5.0)
    window_samples = int(window_size_s * sr)
    hop_samples = int(HOP_SIZE_S * sr)
    n_frames = max(1, (len(audio) - window_samples) // hop_samples + 1)

    log.debug(
        "%.1fs audio → %d windows (%.1fs window, %.1fs hop, threshold=%.2f)",
        len(audio) / sr,
        n_frames,
        window_size_s,
        HOP_SIZE_S,
        threshold,
    )

    sources: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    species_list: list[str] = []
    confidences: list[float] = []
    ranks: list[int] = []

    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        if end > len(audio):
            break

        outputs = model.embed(audio[start:end])
        logits_arr: np.ndarray = outputs.logits[logits_key]
        while logits_arr.ndim > 1:
            logits_arr = logits_arr[0]

        probs: np.ndarray = expit(logits_arr)  # sigmoid, independent per class

        # Only keep species above threshold, ranked by confidence
        above = np.where(probs >= threshold)[0]
        if len(above) == 0:
            continue
        above_sorted = above[np.argsort(probs[above])[::-1]]

        t_start = round(offset_s + start / sr, 2)
        t_end = round(offset_s + end / sr, 2)

        for rank, idx in enumerate(above_sorted, start=1):
            sources.append(source_name)
            starts.append(t_start)
            ends.append(t_end)
            species_list.append(model.class_list.classes[idx])
            confidences.append(float(probs[idx]))
            ranks.append(rank)

    log.debug("Emitted %d detection rows from %d windows", len(sources), n_frames)

    return pl.DataFrame(
        {
            "source": sources,
            "time_start_s": starts,
            "time_end_s": ends,
            "species": species_list,
            "confidence": confidences,
            "rank": ranks,
        }
    )
