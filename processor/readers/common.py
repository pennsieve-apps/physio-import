"""Helpers shared by the readers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from processor.model import Signal

TIME_NAMES = ("time", "t", "times", "time_s", "time (s)", "time [s]", "time(s)", "seconds", "sec", "secs", "timestamp",
              "timestamps", "time_sec", "elapsed time", "elapsed_time")
TIME_UNIT = {"s": 1.0, "sec": 1.0, "secs": 1.0, "seconds": 1.0, "ms": 1e-3, "msec": 1e-3, "min": 60.0, "mins": 60.0,
             "minutes": 60.0, "h": 3600.0, "hr": 3600.0, "hours": 3600.0, "us": 1e-6}
RATE_NAMES = ("fs", "srate", "sfreq", "sampling_rate", "samplingrate", "sample_rate", "samplerate", "sampling_frequency",
              "samplingfrequency", "frequency_hz", "rate", "sr", "freq", "fsamp", "sample_frequency")


@dataclass
class Ctx:
    """What a reader may need from the run: parameters and the user's layout files."""
    time_column: str = ""
    sampling_rate: float = 0.0
    layouts: list = field(default_factory=list)   # (path, dict) of layout.yaml files
    log: object = None
    input_root: object = None                     # INPUT_DIR (the top for BIDS sidecar inheritance)


def split_name_unit(col: str) -> tuple[str, str]:
    """'ECG (mV)' / 'ECG [mV]' / 'ECG, mV' -> ('ECG', 'mV'); otherwise (col, '')."""
    s = str(col).strip()
    m = re.match(r"^(.*?)[\s_]*[\(\[]\s*([^\)\]]{1,12})\s*[\)\]]\s*$", s)
    if m and m.group(1):
        return m.group(1).strip(), m.group(2).strip()
    return s, ""


def time_scale(col: str) -> float | None:
    """Seconds per unit of a time column named like 'Time (ms)'; None when the name is not a time column."""
    name, unit = split_name_unit(col)
    low = name.strip().lower()
    if low not in TIME_NAMES and str(col).strip().lower() not in TIME_NAMES:
        return None
    return TIME_UNIT.get(unit.lower(), 1.0) if unit else 1.0


def segments_from_times(t: np.ndarray, rate_hint: float = 0.0) -> tuple[float, list[tuple[int, int]], bool]:
    """Rate and contiguous runs [(start index, stop index)] of a time vector (s); a gap is a step longer than
    1.5 sample intervals. Returns (rate, runs, regular) where regular=False means the times inside a run
    deviate from start + i/rate by more than half a sample (then explicit timestamps are kept)."""
    t = np.asarray(t, dtype=float)
    n = t.size
    if n < 2:
        return (rate_hint or 1.0), [(0, n)], True
    d = np.diff(t)
    pos = d[d > 0]
    if pos.size == 0:
        raise ValueError("the time column does not increase")
    dt = float(np.median(pos))
    breaks = np.flatnonzero((d > 1.5 * dt) | (d <= 0)) + 1
    edges = [0, *breaks.tolist(), n]
    runs = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    # rate from the longest run's span (more precise than the median step of rounded times)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    rate = (b - a - 1) / (t[b - 1] - t[a]) if b - a > 1 and t[b - 1] > t[a] else 1.0 / dt
    if rate_hint:
        rate = rate_hint
    regular = True
    for a, b in runs:
        k = np.arange(b - a)
        if b - a > 1 and np.max(np.abs(t[a:b] - (t[a] + k / rate))) > 0.5 / rate:
            regular = False
            break
    return float(rate), runs, regular


def nice_rate(rate: float) -> float:
    """Snap a rate estimated from rounded time stamps to a near round number (999.9999 -> 1000)."""
    for r in (round(rate), round(rate, 1), round(rate, 3)):
        if r and abs(r - rate) / rate < 1e-6:
            return float(r)
    return float(rate)


def array_signal(name: str, arrays: list[np.ndarray], rate: float, starts: list[float], unit: str = "",
                 leaf: str = "", times: list[np.ndarray] | None = None, site: str = "") -> Signal:
    """A Signal over in-memory arrays (one per segment)."""
    segs = [(float(s), int(a.shape[0])) for s, a in zip(starts, arrays)]
    dtype = np.result_type(*[a.dtype for a in arrays]) if arrays else np.dtype(float)

    def read(seg, a, b, _arrs=arrays):
        return np.asarray(_arrs[seg][a:b])

    seg_times = None
    if times is not None:
        def seg_times(seg, a, b, _t=times):
            return np.asarray(_t[seg][a:b], dtype=float)
    return Signal(name=name, unit=unit, rate=float(rate), segments=segs, read=read, dtype=np.dtype(dtype),
                  leaf=leaf, segment_times=seg_times, site=site)


def numeric_array(x) -> np.ndarray | None:
    try:
        a = np.asarray(x)
    except Exception:  # noqa: BLE001
        return None
    if a.dtype.kind in "iuf" and a.size:
        return a
    if a.dtype.kind == "b":
        return a.astype(np.uint8)
    return None
