"""
MATLAB .mat files (v4/v5/v7 with SciPy, v7.3 HDF5-based with h5py), read through pymatreader
(Dirk Gütlin and Thomas Hartmann, BSD-2-Clause), which gives both as plain dictionaries.

Layouts recognised without help:
- LabChart "Export > MATLAB" files (data, datastart, dataend, samplerate, titles, unittext,
  unittextmap, blocktimes, com, comtext): each block is a segment, comments are markers;
- Spike2 "Export As > MATLAB" files: one struct per channel (title, units, interval, start,
  values; event and marker channels with times, codes, text become markers);
- a sampling rate (fs, Fs, srate, sfreq, sampling_rate, SampleRate, ...) or a time vector (t,
  time, Time, timestamps) next to signal vectors of the same length.
Anything else needs a layout.yaml (readers/layout.py). A .mat file is loaded whole (MATLAB v5
cannot be read by slice), so a very large .mat needs the memory to hold it.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from processor.model import Marker, SourceRecording, Skip
from processor.readers import layout as L
from processor.readers.common import TIME_NAMES, array_signal, nice_rate, numeric_array, segments_from_times

MAT_RATE_NAMES = {"fs", "srate", "sfreq", "sampling_rate", "samplingrate", "sample_rate", "samplerate",
                  "sampling_frequency", "samplingfrequency", "fsamp", "sr", "fs_hz", "samplingrate_hz"}


def _load(path: Path) -> dict:
    from pymatreader import read_mat

    d = read_mat(str(path))
    return {k: v for k, v in d.items() if not k.startswith("__")}


def _strings(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x.strip()]
    a = np.asarray(x, dtype=object)
    if a.ndim == 2 and all(isinstance(c, str) and len(c) <= 1 for c in a.reshape(-1)):
        return ["".join(row).strip() for row in a]
    return [str(s).strip() for s in a.reshape(-1)]


def _get_path(d: dict, p: str):
    cur = d
    for part in p.split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"{p!r} not found")
    return cur


def read(path: Path, fmt: str, ctx) -> list:
    d = _load(path)
    rec = SourceRecording(path=path, source_format="mat")
    lay = L.layout_for(ctx.layouts, path)
    if lay is not None:
        L.apply(lambda p: _get_path(d, p), lay, rec, path.name)
        return [rec]
    if {"data", "datastart", "dataend", "samplerate"} <= set(d):
        _labchart(d, rec)
        return [rec]
    if _spike2(d, rec):
        return [rec]
    if _simple(d, rec, ctx):
        return [rec]
    names = ", ".join(f"{k} {np.shape(v)}" for k, v in list(d.items())[:12])
    raise Skip(f"MATLAB layout not recognised (variables: {names}): name the sampling rate 'fs' or the time "
               "vector 't', or describe the file in a layout.yaml")


def _labchart(d: dict, rec: SourceRecording) -> None:
    data = np.asarray(d["data"]).reshape(-1)
    ds = np.atleast_2d(np.asarray(d["datastart"], dtype=float))
    de = np.atleast_2d(np.asarray(d["dataend"], dtype=float))
    sr = np.atleast_2d(np.asarray(d["samplerate"], dtype=float))
    nch, nbl = ds.shape
    titles = _strings(d.get("titles")) or [f"Channel {i + 1}" for i in range(nch)]
    units = _strings(d.get("unittext"))
    umap = np.atleast_2d(np.asarray(d.get("unittextmap", np.full((nch, nbl), -1)), dtype=float))
    bt = np.asarray(d.get("blocktimes", []), dtype=float).reshape(-1)
    fso = np.atleast_2d(np.asarray(d.get("firstsampleoffset", np.zeros((nch, nbl))), dtype=float))

    def datenum(x):
        return datetime(1, 1, 1) + timedelta(days=float(x) - 367)

    if bt.size:
        rec.start_time = datenum(bt[0])
        block_start = [(datenum(b) - rec.start_time).total_seconds() for b in bt]
    else:
        block_start = []
        t = 0.0
        for b in range(nbl):
            block_start.append(t)
            lens = [(de[c, b] - ds[c, b] + 1) / sr[c, b] for c in range(nch) if ds[c, b] > 0 and sr[c, b] > 0]
            t += (max(lens) if lens else 0) + 1.0
        if nbl > 1:
            rec.notes.append("no blocktimes: LabChart blocks placed 1 s apart")
    for c in range(nch):
        arrays, starts, rate = [], [], None
        unit = ""
        for b in range(nbl):
            if ds[c, b] <= 0 or de[c, b] < ds[c, b]:
                continue
            a0, a1 = int(ds[c, b]) - 1, int(de[c, b])
            arrays.append(data[a0:a1])
            starts.append(block_start[b] + (fso[c, b] if fso.shape == ds.shape else 0.0))
            r = float(sr[c, b])
            if rate is not None and abs(r - rate) > 1e-9:
                rec.notes.append(f"{titles[c]}: sample rate changes between blocks; block {b + 1} skipped")
                arrays.pop()
                starts.pop()
                continue
            rate = r
            k = int(umap[c, b]) if umap.shape == ds.shape else -1
            if k >= 1 and k <= len(units) and not unit:
                unit = units[k - 1]
        if arrays:
            rec.signals.append(array_signal(titles[c] if c < len(titles) else f"Channel {c + 1}", arrays, rate, starts,
                                            unit=unit))
    com = d.get("com")
    if com is not None and np.size(com):
        com = np.atleast_2d(np.asarray(com, dtype=float))
        texts = _strings(d.get("comtext"))
        tick = np.asarray(d.get("tickrate", sr.max()), dtype=float).reshape(-1)
        for row in com:
            ch, blk, pos, _typ, ti = (int(row[0]), int(row[1]) - 1, float(row[2]), row[3], int(row[4]))
            if not (0 <= blk < nbl):
                continue
            tr = tick[blk] if blk < tick.size else tick[0]
            text = texts[ti - 1] if 1 <= ti <= len(texts) else ""
            rec.markers.append(Marker(block_start[blk] + pos / tr, None, text, "marker",
                                      {"channel": ch} if ch > 0 else {}))
    rec.notes.append(f"LabChart MATLAB export: {nch} channel(s), {nbl} block(s)")


def _spike2(d: dict, rec: SourceRecording) -> bool:
    found = False
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        keys = {x.lower() for x in v}
        title = str(v.get("title", k)).strip() or k
        if {"values", "interval"} <= keys:
            vals = numeric_array(v["values"])
            if vals is None:
                continue
            rate = 1.0 / float(v["interval"])
            start = float(v.get("start", 0.0) or 0.0)
            rec.signals.append(array_signal(title, [vals.reshape(-1)], nice_rate(rate), [start],
                                            unit=str(v.get("units", "")).strip()))
            found = True
        elif "times" in keys and "values" not in keys:
            times = np.asarray(v["times"], dtype=float).reshape(-1)
            texts = _strings(v.get("text")) if v.get("text") is not None else []
            codes = np.asarray(v.get("codes", []))
            for i, t in enumerate(times):
                lab = texts[i] if i < len(texts) and texts[i] else title
                if not texts and codes.size:
                    c = np.atleast_2d(codes)[i] if np.atleast_2d(codes).shape[0] > i else None
                    if c is not None and np.any(c):
                        lab = f"{title}: code {int(c.reshape(-1)[0])}"
                rec.markers.append(Marker(float(t), None, lab, "marker", {"channel_name": title}))
    if found:
        rec.notes.append("Spike2 MATLAB export")
    return found


def _flatten(d: dict, prefix: str = "", depth: int = 0) -> dict:
    out = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict) and depth < 2:
            out.update(_flatten(v, name + "/", depth + 1))
        else:
            out[name] = v
    return out


def _simple(d: dict, rec: SourceRecording, ctx) -> bool:
    flat = _flatten(d)
    rate = None
    time = None
    for k, v in flat.items():
        leaf = k.split("/")[-1].lower()
        a = numeric_array(v)
        if a is None:
            continue
        if leaf in MAT_RATE_NAMES and a.size == 1 and rate is None:
            rate = float(a.reshape(-1)[0])
        elif leaf in TIME_NAMES and a.ndim <= 2 and min(a.shape or (0,)) <= 1 and a.size > 10 and time is None:
            time = (k, a.reshape(-1).astype(float))
    if rate is None and ctx.sampling_rate:
        rate = ctx.sampling_rate
    if rate is None and time is None:
        return False
    n = time[1].size if time else None
    sigs = []
    for k, v in flat.items():
        if time and k == time[0]:
            continue
        a = numeric_array(v)
        if a is None or a.size < 100:
            continue
        if a.ndim == 1 or (a.ndim == 2 and min(a.shape) == 1):
            x = a.reshape(-1)
            if n is None or x.size == n:
                sigs.append((k, x))
        elif a.ndim == 2:
            ax = None
            if n is not None:
                ax = 0 if a.shape[0] == n else (1 if a.shape[1] == n else None)
            else:
                ax = 0 if a.shape[0] >= a.shape[1] else 1
            if ax is None:
                continue
            m = a if ax == 0 else a.T
            for j in range(m.shape[1]):
                sigs.append((f"{k}/{j + 1}", np.ascontiguousarray(m[:, j])))
    if not sigs:
        return False
    if time is not None:
        t = time[1]
        r, runs, regular = segments_from_times(t)
        r = nice_rate(r)
        for k, x in sigs:
            rec.signals.append(array_signal(k, [x[a:b] for a, b in runs], r, [t[a] - t[0] for a, _ in runs],
                                            leaf=k.split("/")[-1],
                                            times=None if regular else [t[a:b] - t[0] for a, b in runs]))
    else:
        for k, x in sigs:
            rec.signals.append(array_signal(k, [x], rate, [0.0], leaf=k.split("/")[-1]))
    return True
