"""
User layout files (layout.yaml) for HDF5 and MATLAB files without a standard layout.

    files: "*.mat"            # which files this layout is for (glob on the name or relative path); default all
    rate: 1000                # samples per second, or the name of the variable/dataset that holds it ("Fs")
    time: t                   # or a time vector (seconds; time_unit: ms for milliseconds)
    start_time: "2024-01-25T14:36:37"   # optional wall-clock start
    signals:                  # variables / datasets ('/' between struct fields or HDF5 groups)
      - path: data            # 1-D, or 2-D samples x channels (or channels x samples: the longer axis is time)
        names: [ECG, BP]      # one name per column (default: path/column number)
        units: [mV, mmHg]
      - path: resp
        name: RESP
        unit: a.u.
        rate: 100             # a signal with its own rate
    trials:                   # 2-D signals shaped trials x samples: one segment per trial
      starts: tst             # trial start times in s (default: back to back with gap_s between them)
      gap_s: 1
      parameters: [Ist, Qst]  # per-trial vectors copied to the stimulation_source table
    markers:
      times: event_times      # marker times in s
      labels: event_labels    # optional texts
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np

from processor.model import Marker, SourceRecording
from processor.readers.common import array_signal, numeric_array


def load_layouts(paths: list[Path]) -> list[tuple[Path, dict]]:
    import yaml

    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        docs = doc if isinstance(doc, list) else [doc]
        for d in docs:
            if not isinstance(d, dict) or not d.get("signals"):
                raise ValueError(f"{p.name}: a layout needs a 'signals' list")
            out.append((p, d))
    return out


def layout_for(layouts: list[tuple[Path, dict]], path: Path) -> dict | None:
    for lp, d in layouts:
        g = str(d.get("files", "*"))
        try:
            rel = str(path.relative_to(lp.parent))
        except ValueError:
            rel = path.name
        if fnmatch.fnmatch(path.name, g) or fnmatch.fnmatch(rel, g):
            return d
    return None


def _scalar(get, v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    a = numeric_array(get(str(v)))
    if a is None or a.size != 1:
        raise ValueError(f"layout: {v!r} is not a number or a scalar variable")
    return float(a.reshape(-1)[0])


def apply(get, layout: dict, rec: SourceRecording, path_label: str) -> None:
    """Fill rec from a layout. `get(name)` returns the array (or scalar) at a variable/dataset path."""
    tunit = {"s": 1.0, "ms": 1e-3, "min": 60.0}.get(str(layout.get("time_unit", "s")).lower(), 1.0)
    trials = layout.get("trials") or None
    base_rate = _scalar(get, layout["rate"]) if layout.get("rate") is not None else None
    base_time = None
    if layout.get("time") is not None:
        base_time = np.asarray(numeric_array(get(str(layout["time"]))), dtype=float).reshape(-1) * tunit
    if base_rate is None and base_time is None and not all(s.get("rate") for s in layout["signals"]):
        raise ValueError("layout: give 'rate' or 'time' (or a rate per signal)")
    if layout.get("start_time"):
        from datetime import datetime
        rec.start_time = datetime.fromisoformat(str(layout["start_time"]))
    n_trials = None
    pending = []
    for spec in layout["signals"]:
        p = str(spec["path"])
        a = numeric_array(get(p))
        if a is None:
            raise ValueError(f"layout: {p!r} is not a numeric array in {path_label}")
        rate = _scalar(get, spec["rate"]) if spec.get("rate") is not None else base_rate
        if trials:
            a = np.atleast_2d(a)
            n_trials = a.shape[0] if n_trials is None else n_trials
            if a.shape[0] != n_trials:
                raise ValueError(f"layout: {p!r} has {a.shape[0]} trials, another signal {n_trials}")
            if rate is None:
                t = base_time
                rate = (t.size - 1) / (t[-1] - t[0])
            pending.append((spec, p, a, rate))
            continue
        if a.ndim == 1:
            cols = [(str(spec.get("name") or p), a, str(spec.get("unit", "")))]
        else:
            if a.ndim != 2:
                raise ValueError(f"layout: {p!r} has {a.ndim} dimensions")
            if a.shape[0] < a.shape[1]:
                a = a.T
            names = spec.get("names") or [f"{p}/{j + 1}" for j in range(a.shape[1])]
            units = spec.get("units") or [str(spec.get("unit", ""))] * a.shape[1]
            idx = [int(spec["column"])] if spec.get("column") is not None else range(a.shape[1])
            cols = [(str(names[j]) if j < len(names) else f"{p}/{j + 1}", a[:, j], str(units[j]) if j < len(units) else "")
                    for j in idx]
        for name, x, unit in cols:
            if rate is not None:
                rec.signals.append(array_signal(name, [np.ascontiguousarray(x)], rate, [0.0], unit=unit,
                                                leaf=name.split("/")[-1]))
            else:
                from processor.readers.common import nice_rate, segments_from_times
                t = base_time
                if t.size != x.shape[0]:
                    raise ValueError(f"layout: {name!r} has {x.shape[0]} samples, the time vector {t.size}")
                r, runs, regular = segments_from_times(t)
                r = nice_rate(r)
                rec.signals.append(array_signal(name, [x[a0:b0] for a0, b0 in runs], r,
                                                [t[a0] - t[0] for a0, _ in runs], unit=unit, leaf=name.split("/")[-1],
                                                times=None if regular else [t[a0:b0] - t[0] for a0, b0 in runs]))
    if pending:
        # one start time per trial for every signal: given, or back to back (longest trial + gap_s)
        if trials.get("starts") is not None:
            starts = np.asarray(numeric_array(get(str(trials["starts"]))), float).reshape(-1) * tunit
            if starts.size != n_trials:
                raise ValueError(f"layout: trials.starts has {starts.size} values for {n_trials} trials")
        else:
            longest = max(a.shape[1] / r for _, _, a, r in pending)
            starts = np.arange(n_trials) * (longest + float(trials.get("gap_s", 1.0)))
            rec.notes.append("trials placed back to back (no trial start times in the file)")
        for spec, p, a, rate in pending:
            name = str(spec.get("name") or p)
            rec.signals.append(array_signal(name, [a[i] for i in range(n_trials)], rate, list(starts),
                                            unit=str(spec.get("unit", "")), leaf=name.split("/")[-1]))
    if trials and trials.get("parameters"):
        sig0 = rec.signals[0]
        rows = []
        params = {str(k): np.asarray(get(str(k))).reshape(-1) for k in trials["parameters"]}
        for i, (st, n) in enumerate(sig0.segments):
            row = {"start_time": float(st), "stop_time": float(st + n / sig0.rate)}
            for k, v in params.items():
                row[k.split("/")[-1]] = v[i].item() if i < v.size and hasattr(v[i], "item") else (v[i] if i < v.size else None)
            rows.append(row)
        rec.stim_table = rows
    mk = layout.get("markers")
    if mk:
        times = np.asarray(numeric_array(get(str(mk["times"]))), float).reshape(-1) * tunit
        labels = get(str(mk["labels"])) if mk.get("labels") else None
        labels = [str(x) for x in np.asarray(labels).reshape(-1)] if labels is not None else ["marker"] * times.size
        for t, lab in zip(times, labels):
            rec.markers.append(Marker(float(t), None, lab, "marker"))
    rec.notes.append("read with a user layout.yaml")
