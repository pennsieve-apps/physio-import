"""
Generic HDF5 with h5py (Andrew Collette and contributors, BSD-3-Clause).

HDF5 has no standard layout for physiology, so three cases:
- an NWB file saved as .h5/.hdf5: read as NWB;
- the common "one group per trial, column datasets" layout: datasets with a `_column_names`
  (or `column_names`/`columns`) attribute and a rate attribute (`frequency_hz`, `sampling_rate`,
  `rate`, `fs`, ...), or a time column (`timestamps`, `time`, `t`). Groups of trials under one
  parent group form one recording (each trial a segment at its own time); the attributes of each
  trial group (stimulation parameters, for example) are copied to `stimulation_source` as found;
- anything else needs a layout.yaml (see readers/layout.py).
Datasets are read by slice, one trial and one column at a time.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from processor.model import Signal, SourceRecording, Skip
from processor.readers import layout as L
from processor.readers.common import RATE_NAMES, TIME_NAMES, nice_rate

COLS = ("_column_names", "column_names", "columns", "col_names")


def is_nwb(path: Path) -> bool:
    import h5py

    try:
        with h5py.File(path, "r") as f:
            return "nwb_version" in f.attrs or ("general" in f and "acquisition" in f)
    except OSError:
        return False


def _attr_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def _columns(ds) -> list[str] | None:
    for k in COLS:
        if k in ds.attrs:
            v = ds.attrs[k]
            return [_attr_str(x) for x in np.asarray(v).reshape(-1)]
    return None


def _rate_attr(obj) -> float | None:
    for k, v in obj.attrs.items():
        if k.lower() in RATE_NAMES:
            try:
                return float(np.asarray(v).reshape(-1)[0])
            except (TypeError, ValueError):
                return None
    return None


def _plain(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else ", ".join(_attr_str(x) for x in v.reshape(-1))
    return v


def read(path: Path, fmt: str, ctx) -> list:
    import h5py

    lay = L.layout_for(ctx.layouts, path)
    f = h5py.File(path, "r")
    if lay is not None:
        rec = SourceRecording(path=path, source_format="hdf5", close=f.close)

        def get(p):
            obj = f[p]
            return obj[()] if isinstance(obj, h5py.Dataset) else None
        L.apply(get, lay, rec, path.name)
        return [rec]

    # find column datasets: trial group -> [datasets]
    trials: dict[str, list] = {}

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset) or obj.ndim not in (1, 2) or obj.dtype.kind not in "iuf":
            return
        cols = _columns(obj)
        if cols is None and _rate_attr(obj) is None:
            return
        trials.setdefault(obj.parent.name, []).append(obj)

    f.visititems(visit)
    if not trials:
        f.close()
        raise Skip("HDF5 layout not recognised (no datasets with column names or a sampling-rate attribute): "
                   "describe it in a layout.yaml (paths of time, signals, rates, units)")
    by_parent: dict[str, list[str]] = {}
    for g in trials:
        parent = g.rsplit("/", 1)[0] or "/"
        by_parent.setdefault(parent, []).append(g)
    recs = []
    for parent, groups in sorted(by_parent.items()):
        if len(groups) == 1:
            parent = groups[0]
        recs.append(_recording(f, path, parent, groups))
    if recs:
        recs[0].close = f.close
    return recs


def _recording(f, path: Path, parent: str, groups: list[str]) -> SourceRecording:
    rec = SourceRecording(path=path, source_format="hdf5", key=parent.strip("/"))
    # per trial: datasets, their columns, rate, first time
    info = []
    for g in groups:
        for ds in sorted(_dsets(f, g), key=lambda d: d.name):
            cols = _columns(ds) or ([ds.name.rsplit("/", 1)[-1]] if ds.ndim == 1 else
                                    [f"{j + 1}" for j in range(ds.shape[1])])
            rate = _rate_attr(ds)
            tcol = next((j for j, c in enumerate(cols) if c.strip().lower() in TIME_NAMES), None)
            t0 = None
            if tcol is not None and ds.shape[0]:
                t0 = float(ds[0, tcol] if ds.ndim == 2 else ds[0])
                if rate is None and ds.shape[0] > 1:
                    t_end = float(ds[ds.shape[0] - 1, tcol] if ds.ndim == 2 else ds[-1])
                    rate = nice_rate((ds.shape[0] - 1) / (t_end - t0)) if t_end > t0 else None
            if rate is None:
                continue
            info.append({"group": g, "ds": ds, "cols": cols, "rate": rate, "tcol": tcol, "t0": t0})
    if not info:
        raise Skip(f"HDF5 group {parent}: no dataset with a sampling rate or a time column")
    have_t = all(i["t0"] is not None for i in info)
    order = sorted(groups, key=lambda g: min((i["t0"] for i in info if i["group"] == g and i["t0"] is not None),
                                             default=0.0) if have_t else g)
    t_ref = min(i["t0"] for i in info) if have_t else 0.0
    if have_t and t_ref > 1e8:  # Unix time stamps: the recording start is an instant (UTC)
        rec.start_time = datetime.fromtimestamp(t_ref, tz=timezone.utc)
        rec.notes.append("start time from Unix time stamps (UTC)")
    if not have_t and len(order) > 1:
        rec.notes.append("no time column: trials placed back to back, 1 s apart (relative times are not known)")
    # segment start per trial when there are no time stamps: back to back
    trial_start = {}
    t = 0.0
    for g in order:
        trial_start[g] = t
        durs = [i["ds"].shape[0] / i["rate"] for i in info if i["group"] == g]
        t += (max(durs) if durs else 0.0) + 1.0
    # channels: (dataset name within trial, column) -> per-trial (dataset, start)
    chans: dict[tuple[str, int], list] = {}
    meta: dict[tuple[str, int], tuple[str, float]] = {}
    for i in info:
        dname = i["ds"].name.rsplit("/", 1)[-1]
        for j, c in enumerate(i["cols"]):
            if j == i["tcol"]:
                continue
            key = (dname, j)
            start = (i["t0"] - t_ref) if have_t else trial_start[i["group"]]
            chans.setdefault(key, []).append((order.index(i["group"]), i["ds"], start, j, i["tcol"]))
            meta[key] = (c, i["rate"])
    for key, parts in chans.items():
        parts.sort(key=lambda p: p[0])
        col, rate = meta[key]
        segs, pieces, n_trim = _merge_overlaps(parts, rate)
        if n_trim:
            rec.notes.append(f"{col}: {n_trim} trial window(s) overlap the previous one with identical samples; "
                             "the repeated samples are written once")
        dtype = parts[0][1].dtype
        seg_times = None
        if have_t and all(p[4] is not None for p in parts):
            def seg_times(seg, a, b, _p=pieces, _tc=parts[0][4], _ref=t_ref):
                return _seg_read(_p[seg], a, b, _tc) - _ref

        def rd(seg, a, b, _p=pieces):
            out = []
            for ds, j, src0, dst0, n in _p[seg]:
                lo, hi = max(a, dst0), min(b, dst0 + n)
                if lo < hi:
                    s0 = src0 + lo - dst0
                    out.append(np.asarray(ds[s0:s0 + hi - lo, j] if ds.ndim == 2 else ds[s0:s0 + hi - lo]))
            return np.concatenate(out) if out else np.zeros(0, dtype=_p[seg][0][0].dtype)

        unit = ""
        ds0 = parts[0][1]
        for k in ("units", "unit"):
            if k in ds0.attrs:
                u = [_attr_str(x) for x in np.asarray(ds0.attrs[k]).reshape(-1)]
                unit = u[key[1]] if len(u) > key[1] else (u[0] if len(u) == 1 else "")
        rec.signals.append(Signal(name=f"{key[0]}/{col}" if col != key[0] else col, unit=unit, rate=float(rate),
                                  segments=segs, read=rd, dtype=np.dtype(dtype), leaf=col,
                                  segment_times=seg_times))
    # per-trial attributes -> stimulation_source (exactly as found)
    rows = []
    for g in order:
        attrs = {k: _plain(v) for k, v in f[g].attrs.items()}
        segs = [(((i["t0"] - t_ref) if have_t else trial_start[g]), i["ds"].shape[0] / i["rate"])
                for i in info if i["group"] == g]
        st = min(s for s, _ in segs)
        sp = max(s + d for s, d in segs)
        if attrs:
            rows.append({"start_time": float(st), "stop_time": float(sp), "trial_group": g.rsplit("/", 1)[-1], **attrs})
    if rows and len(order) > 1:
        rec.stim_table = rows
    subj = parent.strip("/").rsplit("/", 1)[-1]
    m = re.match(r"^(?:sub(?:ject)?)[-_ ]?(.+)$", subj, re.I)
    if m:
        rec.subject_id = m.group(1)
    return rec


def _seg_read(pieces, a, b, col):
    """Samples a:b of a segment made of pieces (column `col`, or each piece's own column when None)."""
    out = []
    for ds, j, src0, dst0, n in pieces:
        lo, hi = max(a, dst0), min(b, dst0 + n)
        if lo < hi:
            s0 = src0 + lo - dst0
            c = j if col is None else col
            out.append(np.asarray(ds[s0:s0 + hi - lo, c] if ds.ndim == 2 else ds[s0:s0 + hi - lo]))
    return np.concatenate(out) if out else np.zeros(0)


def _merge_overlaps(parts, rate):
    """Trials cut from one continuous acquisition often overlap (each window starts before the previous one
    ended). Where the overlapping samples are identical, the repeat is dropped and trials that then touch are
    joined into one segment; overlapping samples that differ are an error (time would run backwards).
    Returns segments [(start, n)], pieces per segment [(dataset, column, source start, segment offset, n)] and
    the number of trimmed overlaps."""
    segs, pieces, n_trim = [], [], 0
    for _, ds, start, j, tcol in parts:
        n = int(ds.shape[0])
        src0 = 0
        if segs:
            s_prev, n_prev = segs[-1]
            end = s_prev + n_prev / rate
            m = int(round((end - start) * rate))
            if m > 0 and tcol is not None:
                # count the overlap on the time stamps themselves (the clock may differ slightly from the nominal
                # rate); the overlap may reach back over several earlier trials of this segment
                k = min(n_prev, m + 50)
                tp = _seg_read(pieces[-1], n_prev - k, n_prev, tcol)
                m = int(tp.size - np.searchsorted(tp, float(ds[0, tcol]) - 0.5 / rate))
            if m > 0:
                m = min(m, n, n_prev)
                a = _seg_read(pieces[-1], n_prev - m, n_prev, None)
                b = np.asarray(ds[0:m, j] if ds.ndim == 2 else ds[0:m])
                if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                    raise ValueError(f"trial windows overlap by {m} samples with different values: cannot order them "
                                     "in time (describe the file in a layout.yaml)")
                src0, n_trim = m, n_trim + 1
                start = start + m / rate
            # a trimmed trial continues the previous one sample for sample; otherwise join when it starts within
            # half a sample of the previous end
            if src0 > 0 or abs((start - end) * rate) < 0.5:
                if n - src0 > 0:
                    pieces[-1].append((ds, j, src0, n_prev, n - src0))
                    segs[-1] = (s_prev, n_prev + n - src0)
                continue
        if n - src0 > 0:
            segs.append((float(start), n - src0))
            pieces.append([(ds, j, src0, 0, n - src0)])
    return segs, pieces, n_trim


def _dsets(f, g):
    import h5py

    return [v for v in f[g].values() if isinstance(v, h5py.Dataset) and v.dtype.kind in "iuf" and v.ndim in (1, 2)
            and (_columns(v) is not None or _rate_attr(v) is not None)]
