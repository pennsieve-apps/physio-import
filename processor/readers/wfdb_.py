"""
WFDB records (PhysioNet) with WFDB for Python (MIT Laboratory for Computational Physiology, MIT
licence): the header (.hea), the signal file(s), and every annotation file of the record (.atr,
.ecg, .qrs, ...) as markers. Digital values are kept with conversion 1/gain and offset
-baseline/gain. A unit is taken as given only when the header states it: WFDB assumes mV when a
header has none, which for an uncalibrated channel (Fantasia's BP) would be wrong.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from processor.model import Marker, Signal, SourceRecording

# Files next to a record that are never annotation files
NOT_ANN = {".hea", ".dat", ".mat", ".csv", ".txt", ".json", ".pdf", ".png", ".edf", ".wav", ".xlsx", ".md", ".html",
           ".zip", ".gz", ".xyz", ".hdf5", ".h5", ".nwb", ".py", ".m", ".abf", ".smr", ".acq", ".tsv"}
# Invalid-sample sentinels per WFDB storage format
INVALID = {"8": -128, "16": -32768, "61": -32768, "80": -128, "160": -32768, "212": -2048, "310": -512, "311": -512,
           "24": -(2 ** 23), "32": -(2 ** 31)}


def _explicit_units(hea: Path) -> list[str]:
    """Units exactly as the header's signal lines state them ('' when a line has none)."""
    units = []
    lines = [ln for ln in hea.read_text(errors="replace").splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        m = re.match(r"^[\d.eE+-]+(?:\([-\d]+\))?(?:/(\S+))?$", parts[2])
        units.append((m.group(1) or "") if m else "")
    return units


def _comments(comments: list[str]) -> dict:
    out = {}
    text = " ".join(comments)
    m = re.search(r"\bage\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        out["age"] = f"P{m.group(1)}Y"
    m = re.search(r"\bsex\s*[:=]\s*([MF])\b", text, re.I)
    if m:
        out["sex"] = m.group(1).upper()
    return out


def read(path: Path, fmt: str, ctx) -> list:
    import wfdb

    base = str(path.with_suffix(""))
    h = wfdb.rdheader(base)
    if getattr(h, "n_seg", None):
        raise ValueError("multi-segment WFDB records are not supported yet (read each segment's .hea)")
    rec = SourceRecording(path=path, source_format="wfdb", label_hint="")
    rec.subject_id = ""
    if h.base_date and h.base_time:
        rec.start_time = datetime.combine(h.base_date, h.base_time)
    rec.notes.extend(f"header comment: {c}" for c in (h.comments or []))
    meta = _comments(h.comments or [])
    rec.age, rec.sex = meta.get("age", ""), meta.get("sex", "")
    units = _explicit_units(path)
    fs = float(h.fs)
    n = int(h.sig_len or 0)
    for i, name in enumerate(h.sig_name or []):
        fmt_i = str(h.fmt[i])
        gain = float(h.adc_gain[i]) or 200.0
        baseline = float(h.baseline[i])
        dtype = np.dtype(np.int16) if fmt_i in ("8", "16", "61", "80", "160", "212", "310", "311") else np.dtype(np.int32)

        def rd(seg, a, b, _i=i, _dt=dtype):
            r = wfdb.rdrecord(base, sampfrom=a, sampto=b, channels=[_i], physical=False)
            return np.asarray(r.d_signal[:, 0], dtype=_dt)

        rec.signals.append(Signal(name=str(name), unit=units[i] if i < len(units) else "", rate=fs,
                                  segments=[(0.0, n)], read=rd, dtype=dtype, conversion=1.0 / gain,
                                  offset=-baseline / gain, invalid=INVALID.get(fmt_i)))
    # annotation files: <record>.<ext>
    for cand in sorted(path.parent.glob(path.stem + ".*")):
        ext = cand.suffix.lower()
        if ext in NOT_ANN or cand.stem != path.stem:
            continue
        try:
            ann = wfdb.rdann(base, cand.suffix[1:])
        except Exception:  # noqa: BLE001 - not an annotation file
            continue
        afs = float(ann.fs or fs)
        aux = ann.aux_note or [""] * len(ann.sample)
        for s, sym, note, ch in zip(ann.sample, ann.symbol, aux, ann.chan if ann.chan is not None else [0] * len(ann.sample)):
            label = str(sym) + (f" {str(note).strip(chr(0)).strip()}" if note and str(note).strip(chr(0)).strip() else "")
            rec.markers.append(Marker(float(s) / afs, None, label, "annotation",
                                      {"annotator": cand.suffix[1:], "channel": int(ch)}))
        rec.notes.append(f"annotations {cand.name}: {len(ann.sample)}")
    return [rec]
