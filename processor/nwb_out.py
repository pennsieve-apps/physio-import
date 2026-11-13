"""
Write one NWB 2.x file per recording with PyNWB/HDMF (Neurodata Without Borders, BSD-3-Clause),
and check it with NWB Inspector (BSD-3-Clause).

Each channel is a TimeSeries in `acquisition`: stored a slice at a time (an HDMF chunk iterator,
so a long or high-rate channel is never in memory whole), gzip-compressed with the shuffle
filter, integers kept as integers with `conversion`/`offset`. Gap-free channels get `rate` and
`starting_time`; trial-by-trial or gapped channels get `timestamps` (shared by channels with the
same rate and segments). Markers go to a TimeIntervals table `markers`; per-trial parameters to
`stimulation_source`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHUNK = 65536          # samples per HDF5 chunk
BUFFER_CHUNKS = 16     # chunks read and written at a time
GZIP_LEVEL = 4


@dataclass
class Plan:
    name: str
    description: str
    unit: str
    conversion: float
    offset: float
    store_dtype: np.dtype
    to_store: object        # callable(raw ndarray) -> stored ndarray
    signal: object          # model.Signal
    comments: str = ""


class _Layout:
    """Global sample index <-> (segment, local index) of a Signal."""

    def __init__(self, sig):
        self.sig = sig
        self.starts = np.cumsum([0] + [n for _, n in sig.segments])

    def read(self, a: int, b: int) -> list[tuple[int, int, int]]:
        out = []
        for s in range(len(self.sig.segments)):
            s0, s1 = self.starts[s], self.starts[s + 1]
            lo, hi = max(a, s0), min(b, s1)
            if lo < hi:
                out.append((s, int(lo - s0), int(hi - s0)))
        return out


def _iterators():
    from hdmf.data_utils import GenericDataChunkIterator

    class SignalIterator(GenericDataChunkIterator):
        def __init__(self, plan: Plan, **kw):
            self.plan = plan
            self.lay = _Layout(plan.signal)
            self.n = plan.signal.n_samples
            super().__init__(**kw)

        def _get_data(self, selection):
            sl = selection[0]
            a, b = sl.start or 0, sl.stop if sl.stop is not None else self.n
            parts = [self.plan.to_store(self.plan.signal.read(s, lo, hi)) for s, lo, hi in self.lay.read(a, b)]
            return np.concatenate(parts).astype(self.plan.store_dtype, copy=False) if parts else \
                np.zeros(0, self.plan.store_dtype)

        def _get_maxshape(self):
            return (self.n,)

        def _get_dtype(self):
            return self.plan.store_dtype

    class TimesIterator(GenericDataChunkIterator):
        def __init__(self, sig, **kw):
            self.sig = sig
            self.lay = _Layout(sig)
            self.n = sig.n_samples
            super().__init__(**kw)

        def _get_data(self, selection):
            sl = selection[0]
            a, b = sl.start or 0, sl.stop if sl.stop is not None else self.n
            parts = []
            for s, lo, hi in self.lay.read(a, b):
                if self.sig.segment_times is not None:
                    # explicit times are seconds from the recording start
                    parts.append(np.asarray(self.sig.segment_times(s, lo, hi), float))
                else:
                    parts.append(self.sig.segments[s][0] + np.arange(lo, hi, dtype=np.float64) / self.sig.rate)
            return np.concatenate(parts) if parts else np.zeros(0)

        def _get_maxshape(self):
            return (self.n,)

        def _get_dtype(self):
            return np.dtype(np.float64)

    return SignalIterator, TimesIterator


def _shapes(n: int, itemsize: int) -> dict:
    chunk = max(1, min(n, CHUNK))
    buf = max(chunk, min(n, chunk * BUFFER_CHUNKS))
    return {"chunk_shape": (chunk,), "buffer_shape": (buf,), "display_progress": False}


def _seg_key(sig) -> tuple:
    return (round(sig.rate, 9), tuple((round(s, 9), n) for s, n in sig.segments), sig.segment_times is not None)


def write(path: Path, rid: str, start: datetime | None, description: str, subject: dict, plans: list[Plan],
          markers: list, stim_table: list[dict] | None, source_name: str) -> None:
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    SignalIterator, TimesIterator = _iterators()
    if start is None:
        sst = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elif start.tzinfo is None:
        sst = start.replace(tzinfo=timezone.utc)
    else:
        sst = start
    nwb = NWBFile(session_description=description, identifier=rid, session_start_time=sst,
                  source_script="physio-import", source_script_file_name=source_name)
    if subject.get("subject_id"):
        kw = {k: v for k, v in subject.items() if v}
        nwb.subject = Subject(**kw)
    shared: dict[tuple, object] = {}
    for p in plans:
        sig = p.signal
        n = sig.n_samples
        data = H5DataIO(SignalIterator(p, **_shapes(n, p.store_dtype.itemsize)), compression="gzip",
                        compression_opts=GZIP_LEVEL, shuffle=True)
        kw = dict(name=p.name, data=data, unit=p.unit, conversion=float(p.conversion), offset=float(p.offset),
                  description=p.description, comments=p.comments or "no comments")
        regular = len(sig.segments) == 1 and sig.segment_times is None
        if regular:
            ts = TimeSeries(rate=float(sig.rate), starting_time=float(sig.segments[0][0]), **kw)
        else:
            key = _seg_key(sig)
            if key in shared:
                ts = TimeSeries(timestamps=shared[key], **kw)
            else:
                tdata = H5DataIO(TimesIterator(sig, **_shapes(n, 8)), compression="gzip", compression_opts=GZIP_LEVEL,
                                 shuffle=True)
                ts = TimeSeries(timestamps=tdata, **kw)
                shared[key] = ts
        nwb.add_acquisition(ts)
    if markers:
        ti = TimeIntervals(name="markers", description="Markers and annotations found in the source file, as found "
                                                       "(label = the original text or annotation code), and clipped "
                                                       "stretches found by physio-import's checks.")
        ti.add_column("label", "marker text or annotation code as in the source")
        ti.add_column("source", "where the row came from: marker = the file's own markers and annotations; qc = a "
                                "stretch physio-import found clipped (trial_type artifact)")
        ti.add_column("trial_type", "marker, annotation or artifact (as in events.tsv)")
        for m in sorted(markers, key=lambda m: m.onset):
            ti.add_row(start_time=float(m.onset), stop_time=float(m.onset + (m.duration or 0.0)), label=m.label,
                       source=(m.extra or {}).get("source", "marker"), trial_type=m.trial_type)
        nwb.add_time_intervals(ti)
    if stim_table:
        ti = TimeIntervals(name="stimulation_source", description=(
            "Per-trial parameters exactly as stored in the source file (names as in the source); start_time and "
            "stop_time are the trial's recorded window, not the stimulation train."))
        cols = []
        for row in stim_table:
            for k in row:
                if k not in ("start_time", "stop_time") and k not in cols:
                    cols.append(k)
        types = {}
        for c in cols:
            vals = [row.get(c) for row in stim_table]
            if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in vals
                   if v is not None) and any(v is not None for v in vals):
                types[c] = float
            else:
                types[c] = str
            ti.add_column(_col_name(c), f"'{c}' as found in the source")
        for row in stim_table:
            vals = {}
            for c in cols:
                v = row.get(c)
                vals[_col_name(c)] = (float(v) if v is not None else math.nan) if types[c] is float else \
                    ("" if v is None else str(v))
            ti.add_row(start_time=float(row["start_time"]), stop_time=float(row["stop_time"]), **vals)
        nwb.add_time_intervals(ti)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def _col_name(c: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
    return s if s and s not in ("start_time", "stop_time", "id", "tags", "timeseries") else f"source_{s}"


# ------------------------------------------------------------------------------------- NWB Inspector
BIG_ARRAY = 5_000_000
INSPECT_CHUNK = 2_000_000
_CHECKS = None


def bounded_checks() -> list:
    """NWB Inspector 0.7.2's check_regular_timestamps reads every timestamp at once (np.diff over the whole
    array); above BIG_ARRAY samples it runs here a chunk at a time with the same result and message (as in the
    nwb-inspect app)."""
    global _CHECKS
    if _CHECKS is not None:
        return _CHECKS
    from nwbinspector import available_checks, register_check
    from nwbinspector._types import InspectorMessage, Severity

    originals = {c.__name__: c for c in list(available_checks)}
    reg_ts = originals["check_regular_timestamps"]

    @register_check(importance=reg_ts.importance, neurodata_type=reg_ts.neurodata_type)
    def check_regular_timestamps(time_series, time_tolerance_decimals: int = 9, gb_severity_threshold: float = 1.0):
        ts = time_series.timestamps
        if ts is None or not hasattr(ts, "shape") or int(ts.shape[0]) <= BIG_ARRAY:
            return reg_ts.__wrapped__(time_series, time_tolerance_decimals, gb_severity_threshold)
        n = int(ts.shape[0])
        step = None
        for a in range(0, n - 1, INSPECT_CHUNK):
            x = np.asarray(ts[a:min(n, a + INSPECT_CHUNK + 1)], dtype=float)
            d = np.unique(np.diff(x).round(decimals=time_tolerance_decimals))
            if len(d) != 1 or (step is not None and d[0] != step):
                return None
            step = d[0]
        if not step:
            return None
        t0, t1 = float(ts[0]), float(ts[1])
        severity = Severity.HIGH if n * ts.dtype.itemsize > gb_severity_threshold * 1e9 else Severity.LOW
        return InspectorMessage(severity=severity, message=(
            "TimeSeries appears to have a constant sampling rate. "
            f"Consider specifying starting_time={t0} and rate={1 / (t1 - t0)} instead of timestamps."))

    _CHECKS = [check_regular_timestamps if name == "check_regular_timestamps" else c for name, c in originals.items()]
    return _CHECKS


def inspect(path: Path) -> list[dict]:
    """NWB Inspector messages (PyNWB schema validation included), as dicts."""
    from nwbinspector import Importance, inspect_nwbfile

    out = []
    try:
        msgs = list(inspect_nwbfile(nwbfile_path=str(path), checks=bounded_checks(), skip_validate=False,
                                    importance_threshold=Importance.BEST_PRACTICE_SUGGESTION))
    except Exception as e:  # noqa: BLE001
        return [{"importance": "ERROR", "check": "inspector", "object": "", "message": f"{type(e).__name__}: {e}"}]
    for m in msgs:
        if m is None:
            continue
        imp = m.importance.name
        if imp == "PYNWB_VALIDATION":
            imp = "CRITICAL"
        text = str(m.message).strip().splitlines()
        out.append({"importance": imp, "check": m.check_function_name or "",
                    "object": f"{m.object_type or ''} {m.object_name or ''}".strip(),
                    "message": text[-1] if imp == "ERROR" and text else " ".join(text)})
    return out
