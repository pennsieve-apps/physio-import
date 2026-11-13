"""
CSV, TSV, text and Excel tables with pandas (BSD-3-Clause) and openpyxl (MIT), and LabChart text
exports.

A table is a recording when it has a time column (`Time`, `time`, `t`, `Time (s)`, `Time (ms)`,
... or the TIME_COLUMN parameter) in seconds, or when SAMPLING_RATE is given; numeric columns
are channels ('ECG (mV)' / 'ECG [mV]' give the unit); empty cells are gaps (kept as NaN, never
zero); text columns are markers (a non-empty cell is a marker at that row's time). Steps in the
time column longer than 1.5 samples split the recording into segments. LabChart "Export > Text
File" (Interval=, ChannelTitle=, UnitName= header lines; '#*' comments) is recognised by its
header.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from processor.model import Marker, SourceRecording, Skip
from processor.readers.common import (array_signal, nice_rate, segments_from_times, split_name_unit,
                                      time_scale)

MIN_ROWS = 100


def _sniff_sep(path: Path) -> str | None:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        head = [fh.readline() for _ in range(5)]
    text = "".join(head)
    if text.count("\t") >= max(1, len(head) - 1):
        return "\t"
    if text.count(",") >= max(1, len(head) - 1):
        return ","
    if text.count(";") >= max(1, len(head) - 1):
        return ";"
    return None


def read(path: Path, fmt: str, ctx) -> list:
    if fmt in ("txt", "csv", "tsv"):
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            head = fh.read(4096)
        if re.search(r"^\s*Interval=", head, re.M) and re.search(r"^\s*ChannelTitle=", head, re.M):
            return [_labchart_txt(path)]
    import pandas as pd

    frames = []
    if fmt == "xlsx":
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Excel file not readable: {e}") from None
        frames = [(name, df) for name, df in sheets.items()]
    else:
        sep = "\t" if fmt == "tsv" else (_sniff_sep(path) or ("," if fmt == "csv" else r"\s+"))
        try:
            df = (pd.read_csv(path, sep=sep, low_memory=False) if len(sep) == 1
                  else pd.read_csv(path, sep=sep, engine="python"))
        except Exception as e:  # noqa: BLE001
            raise Skip(f"not a readable table: {e}") from None
        frames = [("", df)]
    recs, reasons = [], []
    for sheet, df in frames:
        try:
            rec = _from_frame(path, fmt, df, ctx)
        except Skip as e:
            reasons.append(f"{sheet + ': ' if sheet else ''}{e}")
            continue
        if sheet and len(frames) > 1:
            rec.key = sheet
        recs.append(rec)
    if not recs:
        raise Skip("; ".join(reasons))
    return recs


def _from_frame(path: Path, fmt: str, df, ctx) -> SourceRecording:
    import pandas as pd

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if len(df) < MIN_ROWS:
        raise Skip(f"only {len(df)} rows: not a recording")
    tcol, tscale = None, 1.0
    if ctx.time_column:
        if ctx.time_column in df.columns:
            tcol = ctx.time_column
            tscale = time_scale(tcol) or 1.0
    else:
        for c in df.columns:
            s = time_scale(c)
            if s is not None:
                tcol, tscale = c, s
                break
    rec = SourceRecording(path=path, source_format=fmt)
    t = None
    if tcol is not None:
        col = df[tcol]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().mean() > 0.99:
            t = num.to_numpy(dtype=float) * tscale
        else:
            dt = pd.to_datetime(col, errors="coerce")
            if dt.notna().mean() > 0.99:
                rec.start_time = dt.iloc[0].to_pydatetime()
                t = (dt - dt.iloc[0]).dt.total_seconds().to_numpy()
        if t is None:
            raise Skip(f"time column {tcol!r} is neither numbers nor date-times")
        keep = np.isfinite(t)
        if not keep.all():
            df, t = df.loc[keep], t[keep]
    elif not ctx.sampling_rate:
        raise Skip("no time column (Time, t, Time (s), ...) and SAMPLING_RATE not set")
    data_cols, text_cols = [], []
    for c in df.columns:
        if c == tcol:
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        nonnull = df[c].notna().sum()
        if nonnull == 0:
            continue
        if num.notna().sum() >= 0.95 * nonnull:
            data_cols.append((c, num.to_numpy(dtype=float)))
        else:
            text_cols.append(c)
    if not data_cols:
        raise Skip("no numeric columns")
    if t is not None:
        r, runs, regular = segments_from_times(t, ctx.sampling_rate or 0.0)
        rate = nice_rate(r)
        starts = [float(t[a] - t[0]) for a, _ in runs]
        times = None if regular else [t[a:b] - t[0] for a, b in runs]
        if t[0] != 0:
            rec.notes.append(f"time column starts at {t[0]:g} s: times are kept relative to the first row")
    else:
        rate = float(ctx.sampling_rate)
        runs, starts, times = [(0, len(df))], [0.0], None
    for c, x in data_cols:
        name, unit = split_name_unit(c)
        rec.signals.append(array_signal(str(c), [x[a:b] for a, b in runs], rate, starts, unit=unit, leaf=name,
                                        times=times))
    for c in text_cols:
        vals = df[c].to_numpy()
        tt = t if t is not None else np.arange(len(df)) / rate
        for i in np.flatnonzero(pd.notna(vals)):
            s = str(vals[i]).strip()
            if s:
                rec.markers.append(Marker(float(tt[i] - (t[0] if t is not None else 0.0)), None, s, "marker",
                                          {"column": str(c)}))
    return rec


def _labchart_txt(path: Path) -> SourceRecording:
    """LabChart text export: header lines (key=<TAB>values), then rows of time + channels; a new header starts a
    new block. Comments ('#* text') at the end of a row become markers."""
    rec = SourceRecording(path=path, source_format="txt")
    blocks = []
    header: dict = {}
    rows: list[list[float]] = []
    comments: list[tuple[int, str]] = []

    def flush():
        if rows:
            blocks.append((dict(header), np.asarray(rows, dtype=float), list(comments)))
        rows.clear()
        comments.clear()

    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            m = re.match(r"^\s*([A-Za-z][A-Za-z ]*)=\s*(.*)$", line)
            if m:
                if rows:
                    flush()
                header[m.group(1).strip()] = [x.strip() for x in m.group(2).split("\t")]
                continue
            com = None
            if "#" in line:
                line, _, com = line.partition("#")
                com = com.lstrip("*").strip()
            parts = line.replace(",", ".").split("\t") if "\t" in line else line.split()
            vals = []
            for p in parts:
                p = p.strip()
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(np.nan)
            if vals:
                if com:
                    comments.append((len(rows), com))
                rows.append(vals)
    flush()
    if not blocks:
        raise Skip("LabChart text export without data rows")
    hdr0 = blocks[0][0]
    titles = [x for x in hdr0.get("ChannelTitle", []) if x != ""]
    units = hdr0.get("UnitName", [])
    interval = hdr0.get("Interval", ["1"])[0]
    m = re.match(r"([\d.eE+-]+)\s*(\w*)", interval)
    dt = float(m.group(1)) * {"ms": 1e-3, "s": 1.0, "us": 1e-6}.get((m.group(2) or "s").lower(), 1.0)
    rate = nice_rate(1.0 / dt)
    width = max(b[1].shape[1] for b in blocks)
    has_time = width == len(titles) + 1
    starts, arrays = [], []
    t_acc = 0.0
    for hdr, a, coms in blocks:
        if has_time:
            st = float(a[0, 0]) if not starts else max(float(a[0, 0]), t_acc)
            data = a[:, 1:]
        else:
            st = t_acc
            data = a
        starts.append(st)
        arrays.append(data)
        for i, text in coms:
            rec.markers.append(Marker(st + i / rate, None, text, "marker"))
        t_acc = st + data.shape[0] / rate + 1.0
    for j, title in enumerate(titles):
        u = units[j] if j < len(units) else ""
        rec.signals.append(array_signal(title, [arr[:, j] if j < arr.shape[1] else np.full(arr.shape[0], np.nan)
                                                for arr in arrays], rate, starts, unit="" if u == "*" else u))
    rec.notes.append(f"LabChart text export: {len(titles)} channel(s), {len(blocks)} block(s)")
    return rec

