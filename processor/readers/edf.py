"""
EDF, EDF+ and BDF with pyEDFlib (Holger Nahrstaedt and contributors, BSD-2-Clause).

pyEDFlib is used directly rather than through Neo: Neo's EdfRawIO is itself a wrapper around
pyEDFlib, and pyEDFlib gives slice reads of the stored integers, the per-signal physical
dimension and the EDF+ annotations. Digital values are kept (int16 for EDF, int32 for BDF) with
the header's digital-to-physical gain and offset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from processor.model import Marker, Signal, SourceRecording, Skip


def read(path: Path, fmt: str, ctx) -> list[SourceRecording]:
    import pyedflib

    try:
        f = pyedflib.EdfReader(str(path))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"pyEDFlib could not open the file: {e}") from None
    rec = SourceRecording(path=path, source_format="bdf" if path.suffix.lower() == ".bdf" else "edf")
    rec.close = f.close
    try:
        rec.start_time = f.getStartdatetime()
    except Exception:  # noqa: BLE001
        rec.start_time = None
    try:
        hdr = f.getHeader()
        sex = str(hdr.get("sex") or hdr.get("gender") or "")
        rec.sex = {"male": "M", "female": "F", "m": "M", "f": "F"}.get(sex.strip().lower(), "")
    except Exception:  # noqa: BLE001
        pass
    labels = f.getSignalLabels()
    nsamp = f.getNSamples()
    is_bdf = rec.source_format == "bdf"
    for i, lab in enumerate(labels):
        lab = str(lab).strip()
        if lab.lower().startswith("edf annotations") or lab.lower().startswith("bdf annotations"):
            continue
        n = int(nsamp[i])
        if n == 0:
            continue
        fs = float(f.getSampleFrequency(i))
        dmin, dmax = float(f.getDigitalMinimum(i)), float(f.getDigitalMaximum(i))
        pmin, pmax = float(f.getPhysicalMinimum(i)), float(f.getPhysicalMaximum(i))
        conv = (pmax - pmin) / (dmax - dmin) if dmax != dmin else 1.0
        off = pmin - dmin * conv
        dtype = np.dtype(np.int32 if is_bdf else np.int16)

        def rd(seg, a, b, _i=i, _dt=dtype):
            return np.asarray(f.readSignal(_i, start=a, n=b - a, digital=True), dtype=_dt)

        unit = str(f.getPhysicalDimension(i)).strip()
        try:
            tr = str(f.getTransducer(i)).strip()
        except Exception:  # noqa: BLE001
            tr = ""
        rec.signals.append(Signal(name=lab, unit=unit, rate=fs, segments=[(0.0, n)], read=rd, dtype=dtype,
                                  conversion=conv, offset=off, description=(f"transducer: {tr}" if tr else "")))
    try:
        on, du, tx = f.readAnnotations()
        for o, d, t in zip(on, du, tx):
            d = float(d)
            rec.markers.append(Marker(float(o), d if d > 0 else None, str(t).strip(), "annotation"))
    except Exception:  # noqa: BLE001
        pass
    if not rec.signals:
        f.close()
        raise Skip("no signals in the file (annotations only)")
    return [rec]
