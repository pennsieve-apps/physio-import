"""
BIDS continuous recordings: `*_physio.tsv.gz` and `*_stim.tsv.gz` with their JSON sidecars
(BIDS specification, "Physiological recordings"; pandas for the tables).

- The table has no header; the sidecar gives `Columns`, `SamplingFrequency` and `StartTime`
  (seconds relative to the first sample of the run's neural data, e.g. the first fMRI volume;
  may be negative), and per column `Units` and `Description`.
- Sidecars are merged by the BIDS inheritance principle: every `<entities>_<suffix>.json` in the
  file's folder and the folders above it (up to the dataset root, the folder with
  dataset_description.json) whose entities are a subset of the file's applies; the deeper and the
  more specific one wins key by key.
- All `_recording-<label>` files (and a `_stim` file) of one run are one recording, each at its
  own rate. Recording time 0 is the earliest first sample, so BIDS time t is recording time
  t - min(StartTime); the run's `_events.tsv` (task events) and `_physioevents.tsv.gz` are moved
  onto the same timeline and kept as markers, so everything stays aligned.
- Column types: `cardiac` is an ECG when its description, long name or the recording label says
  ECG/EKG, otherwise a pulse (PPG) signal (MR scanners record cardiac activity with a pulse
  oximeter); `respiratory`/`respiration` is respiration; in a _physio file `trigger` is the
  scanner's volume trigger and is typed `other` (it is not a stimulation); in a _stim file every
  column not otherwise recognised is a stimulation channel. Other columns are typed by name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from processor.model import Marker, SourceRecording, Skip
from processor.readers.common import TIME_NAMES, array_signal

FILE_RX = re.compile(r"^(?P<base>.+?)(?:_recording-(?P<rec>[A-Za-z0-9]+))?_(?P<suffix>physio|stim)\.tsv(?:\.gz)?$")
ROW_CHUNK = 1_000_000


def group_key(path: Path) -> tuple[str, str] | None:
    m = FILE_RX.match(path.name)
    return (str(path.parent), m.group("base")) if m else None


def entities(name: str) -> dict:
    stem = re.sub(r"(\.tsv)?(\.gz)?$|\.json$", "", name)
    parts = stem.split("_")
    return dict(p.split("-", 1) for p in parts[:-1] if "-" in p)


def _dataset_root(path: Path, stop: Path | None) -> Path:
    d = path.parent
    while True:
        if (d / "dataset_description.json").is_file() or d == stop or d.parent == d:
            return d
        d = d.parent


def sidecar(path: Path, suffix: str, input_root: Path | None) -> tuple[dict, list[str]]:
    """The merged sidecar of a physio/stim file (inheritance principle) and the files it came from."""
    ents = entities(path.name)
    root = _dataset_root(path, input_root)
    dirs = []
    d = path.parent
    while True:
        dirs.append(d)
        if d == root or d.parent == d:
            break
        d = d.parent
    meta, used = {}, []
    for d in reversed(dirs):  # top-down: deeper files override
        cands = []
        for j in d.glob(f"*{suffix}.json"):
            jn = j.name
            if not jn.endswith(f"_{suffix}.json") and jn != f"{suffix}.json":
                continue
            je = entities(jn) if "_" in jn else {}
            if all(ents.get(k) == v for k, v in je.items()):
                cands.append((len(je), j))
        for _, j in sorted(cands):
            try:
                meta.update(json.loads(j.read_text(encoding="utf-8")))
            except (OSError, ValueError) as e:
                raise ValueError(f"sidecar {j.name} is not valid JSON: {e}") from None
            used.append(str(j.name))
    return meta, used


def _read_table(path: Path, ncol: int) -> list[np.ndarray]:
    import pandas as pd

    cols: list[list[np.ndarray]] = [[] for _ in range(ncol)]
    reader = pd.read_csv(path, sep="\t", header=None, na_values=["n/a"], chunksize=ROW_CHUNK, compression="infer",
                         dtype=np.float64, engine="c")
    for chunk in reader:
        if chunk.shape[1] != ncol:
            raise ValueError(f"{path.name}: {chunk.shape[1]} columns, the sidecar's Columns names {ncol}")
        for j in range(ncol):
            cols[j].append(chunk.iloc[:, j].to_numpy(dtype=np.float32))
    return [np.concatenate(c) if c else np.zeros(0, np.float32) for c in cols]


def _cardiac_type(meta_col: dict, rec_label: str, file_meta: dict) -> tuple[str, str]:
    text = " ".join(str(x) for x in (meta_col.get("Description", ""), meta_col.get("LongName", ""), rec_label,
                                     file_meta.get("Description", ""), file_meta.get("Type", ""))).lower()
    if re.search(r"ecg|ekg|electrocardio", text):
        return "ecg", "BIDS 'cardiac' column described as ECG"
    if re.search(r"pulse|ppg|oxim|pleth|finger", text):
        return "ppg", "BIDS 'cardiac' column described as a pulse signal"
    return "ppg", "BIDS 'cardiac' column without an ECG description: taken as a pulse (PPG) signal"


def read(path: Path, fmt: str, ctx) -> list:
    import pandas as pd

    key = group_key(path)
    if key is None:
        raise Skip("not a BIDS _physio/_stim file name")
    base = key[1]
    files = sorted(p for p in path.parent.iterdir() if (m := FILE_RX.match(p.name)) and m.group("base") == base)
    input_root = getattr(ctx, "input_root", None)
    parts = []
    for f in files:
        m = FILE_RX.match(f.name)
        meta, used = sidecar(f, m.group("suffix"), input_root)
        missing = [k for k in ("Columns", "SamplingFrequency") if k not in meta]
        if missing:
            raise Skip(f"{f.name}: no {' or '.join(missing)} in any applicable _{m.group('suffix')}.json sidecar "
                       "(BIDS requires one)")
        parts.append((f, m.group("suffix"), m.group("rec") or "", meta, used))
    t0 = min(float(p[3].get("StartTime", 0.0) or 0.0) for p in parts)
    rec = SourceRecording(path=path, source_format="bids-physio")
    ents = entities(path.name)
    rec.entities = {k: v for k, v in ents.items() if k != "recording"}
    lab = [ents["task"]] if "task" in ents else []
    for k in ("acq", "ce", "dir", "run", "echo", "part"):
        if k in ents:
            lab.append(f"{k}{ents[k]}")
    rec.label_hint = "-".join(lab) or "physio"
    multi = len(parts) > 1
    for f, suffix, rlabel, meta, used in parts:
        names = [str(c) for c in meta["Columns"]]
        fs = float(meta["SamplingFrequency"])
        start = float(meta.get("StartTime", 0.0) or 0.0) - t0
        data = _read_table(f, len(names))
        rec.notes.append(f"{f.name}: {len(names)} column(s) at {fs:g} Hz, StartTime {meta.get('StartTime', 0)} s; "
                         f"sidecar(s): {', '.join(used)}")
        for name, x in zip(names, data):
            if name.strip().lower() in TIME_NAMES or name.strip().lower() == "timestamp":
                continue
            cm = meta.get(name) if isinstance(meta.get(name), dict) else {}
            full = f"{suffix}" + (f"-{rlabel}" if rlabel else "") + f"/{name}" if multi else name
            sig = array_signal(full, [x], fs, [start], unit=str(cm.get("Units", "") or ""), leaf=name)
            sig.description = str(cm.get("Description", "") or "")
            low = name.strip().lower()
            if low == "cardiac":
                sig.type_hint, sig.type_note = _cardiac_type(cm, rlabel, meta)
            elif low in ("respiratory", "respiration"):
                sig.type_hint, sig.type_note = "resp", "BIDS respiratory column"
            elif low == "trigger" and suffix == "physio":
                sig.type_hint, sig.type_note = "other", "BIDS scanner trigger (volume timing, not a stimulation)"
            elif suffix == "stim":
                from processor.channels import type_by_name
                if type_by_name(name) is None:
                    sig.type_hint, sig.type_note = "stim", "column of a BIDS _stim file"
            rec.signals.append(sig)
        # physiology events of this file
        ev = f.with_name(f.name.replace(f"_{suffix}.tsv", "_physioevents.tsv"))
        if ev.is_file():
            emeta, _ = sidecar(ev, "physioevents", input_root)
            ecols = [str(c) for c in emeta.get("Columns", ["onset", "duration", "trial_type", "message"])]
            df = pd.read_csv(ev, sep="\t", header=None, names=ecols, na_values=["n/a"], compression="infer")
            src = emeta.get("OnsetSource")
            for _, r in df.iterrows():
                on = float(r["onset"])
                if src and src in names:   # onset is a value of a column of the physio file (a device timestamp)
                    col = data[names.index(src)].astype(np.float64)
                    on = start + float(np.searchsorted(col, on)) / fs
                else:
                    on = on - t0
                lab = str(r.get("message", "") if pd.notna(r.get("message", None)) else r.get("trial_type", "event"))
                rec.markers.append(Marker(on, float(r["duration"]) if "duration" in r and pd.notna(r["duration"]) else None,
                                          lab, "marker", {"channel_name": "physioevents"}))
    # the run's task events (BIDS _events.tsv), on the same timeline
    ev_base = re.sub(r"_recording-[A-Za-z0-9]+", "", base)
    evf = path.parent / f"{ev_base}_events.tsv"
    if evf.is_file():
        df = pd.read_csv(evf, sep="\t", na_values=["n/a"])
        if "onset" in df.columns:
            other = [c for c in df.columns if c not in ("onset", "duration")]
            for _, r in df.iterrows():
                if pd.isna(r["onset"]):
                    continue
                if "trial_type" in df.columns and pd.notna(r["trial_type"]):
                    lab = str(r["trial_type"])
                else:
                    lab = "; ".join(f"{c}={r[c]}" for c in other if pd.notna(r[c]))[:200] or "event"
                du = float(r["duration"]) if "duration" in df.columns and pd.notna(r["duration"]) else None
                rec.markers.append(Marker(float(r["onset"]) - t0, du if du and du > 0 else None, lab, "marker",
                                          {"channel_name": evf.name}))
            rec.notes.append(f"task events from {evf.name}: {len(df)}")
    if t0 != 0:
        rec.notes.append(f"BIDS time 0 (the run's first neural sample) is at {-t0:g} s of this recording")
    if not rec.signals:
        raise Skip("no signal columns")
    return [rec]
