"""
NWB 2.x input with PyNWB (Neurodata Without Borders, BSD-3-Clause): every TimeSeries in
`acquisition` (each column of a 2-D series is a channel), read by slice; Subject metadata;
interval tables as markers; a `stimulation_source` table (or, without one, the `trials` table)
as per-trial parameters. A TimeSeries with timestamps is split into segments at gaps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from processor.metadata import age_iso, sex_code, species_code
from processor.model import Marker, Signal, SourceRecording, Skip

CHUNK = 2_000_000


def _segments_from_timestamps(ts, n: int) -> tuple[float, list[tuple[float, int]], list[int]]:
    """Rate and segments of a timestamps dataset, read a chunk at a time; gap = step > 1.5 x median step."""
    first = np.asarray(ts[: min(n, 10_000)], dtype=float)
    d = np.diff(first)
    dt = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    starts_idx = [0]
    prev_last = None
    for a in range(0, n, CHUNK):
        x = np.asarray(ts[a:min(n, a + CHUNK)], dtype=float)
        if prev_last is not None and x.size and x[0] - prev_last > 1.5 * dt:
            starts_idx.append(a)
        dd = np.diff(x)
        starts_idx.extend((a + 1 + np.flatnonzero(dd > 1.5 * dt)).tolist())
        prev_last = float(x[-1]) if x.size else prev_last
    edges = starts_idx + [n]
    t0 = float(ts[0])
    segs = [(float(ts[a]) - t0, b - a) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return 1.0 / dt, segs, edges[:-1]


def read(path: Path, fmt: str, ctx) -> list:
    from pynwb import NWBHDF5IO, TimeSeries

    io = NWBHDF5IO(str(path), "r", load_namespaces=True)
    nwb = io.read()
    rec = SourceRecording(path=path, source_format="nwb")
    rec.close = io.close
    rec.start_time = nwb.session_start_time
    if nwb.subject is not None:
        s = nwb.subject
        rec.subject_id = str(s.subject_id or "")
        rec.species = species_code(s.species or "")
        rec.sex = sex_code(s.sex or "")
        rec.age = age_iso(s.age or "")
    t_first = None
    for name, obj in sorted(nwb.acquisition.items()):
        if not isinstance(obj, TimeSeries):
            continue
        data = obj.data
        shape = getattr(data, "shape", None) or np.asarray(data).shape
        n = int(shape[0])
        if n == 0:
            continue
        ncol = int(shape[1]) if len(shape) > 1 else 1
        if len(shape) > 2:
            rec.notes.append(f"{name}: {len(shape)}-D data skipped")
            continue
        conv = float(obj.conversion or 1.0)
        off = float(getattr(obj, "offset", 0.0) or 0.0)
        if obj.timestamps is not None:
            rate, segs, starts = _segments_from_timestamps(obj.timestamps, n)
            t0 = float(obj.timestamps[0])
        else:
            rate = float(obj.rate)
            t0 = float(obj.starting_time or 0.0)
            segs, starts = [(0.0, n)], [0]
        t_first = t0 if t_first is None else min(t_first, t0)
        dtype = np.dtype(getattr(data, "dtype", np.float64))
        ch_names = None
        if ncol > 1 and hasattr(obj, "electrodes") and obj.electrodes is not None:
            try:
                tbl = obj.electrodes.table
                rows = obj.electrodes.data[:]
                col = "label" if "label" in tbl.colnames else ("location" if "location" in tbl.colnames else None)
                ch_names = [str(tbl[col][int(i)]) for i in rows] if col else None
            except Exception:  # noqa: BLE001
                ch_names = None
        for j in range(ncol):
            def rd(seg, a, b, _d=data, _j=j, _st=starts, _nc=ncol):
                s = _st[seg]
                return np.asarray(_d[s + a:s + b] if _nc == 1 else _d[s + a:s + b, _j])

            nm = name if ncol == 1 else f"{name}/{ch_names[j] if ch_names else j}"
            sig = Signal(name=nm, unit=str(obj.unit or ""), rate=rate, segments=[(st + t0, k) for st, k in segs],
                         read=rd, dtype=dtype, conversion=conv, offset=off, leaf=name if ncol == 1 else nm.split("/")[-1],
                         description=str(obj.description or ""))
            rec.signals.append(sig)
    if not rec.signals:
        io.close()
        raise Skip("no TimeSeries in the NWB file's acquisition")
    # segment starts relative to the first sample of any series
    t_first = t_first or 0.0
    for sig in rec.signals:
        sig.segments = [(st - t_first, k) for st, k in sig.segments]
    # markers: every interval table except trials / stimulation_source
    stim = None
    for tname, tbl in (nwb.intervals or {}).items():
        df = tbl.to_dataframe()
        if tname in ("stimulation_source",) or (tname == "trials" and stim is None):
            stim = df
            continue
        for _, row in df.iterrows():
            label = str(row.get("label", row.get("tags", tname)))
            dur = float(row["stop_time"] - row["start_time"]) if "stop_time" in row else None
            rec.markers.append(Marker(float(row["start_time"]) - t_first, dur if dur and dur > 0 else None, label,
                                      "marker", {"table": tname}))
    if stim is not None:
        rows = []
        for _, row in stim.iterrows():
            d = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
                 if not isinstance(v, (list, np.ndarray)) and k not in ("tags", "timeseries")}
            d["start_time"] = float(d["start_time"]) - t_first
            d["stop_time"] = float(d["stop_time"]) - t_first
            rows.append(d)
        rec.stim_table = rows
    return [rec]
