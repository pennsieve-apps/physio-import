"""
WAV (scipy.io.wavfile, BSD-3-Clause) and BIOPAC AcqKnowledge .acq (bioread, University of
Wisconsin-Madison Center for Healthy Minds, MIT).

WAV has no channel names and no physical unit: channels are 'channel 1', 'channel 2', ... and
uncalibrated (integer samples kept, full scale = 1). Name them with a channels-map.csv (for
example pattern '^channel 1$', type ecg, path_glob '*.wav').
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from processor.model import Marker, Signal, SourceRecording, Skip

FULL_SCALE = {np.dtype(np.int16): 32768.0, np.dtype(np.int32): 2147483648.0, np.dtype(np.uint8): 128.0}


def read_wav(path: Path, fmt: str, ctx) -> list:
    from scipy.io import wavfile

    try:
        rate, data = wavfile.read(str(path), mmap=True)
    except (ValueError, TypeError):
        rate, data = wavfile.read(str(path))
    data = data if data.ndim == 2 else data.reshape(-1, 1)
    rec = SourceRecording(path=path, source_format="wav")
    dt = np.dtype(data.dtype)
    conv = 1.0 / FULL_SCALE[dt] if dt in FULL_SCALE else 1.0
    off = -1.0 if dt == np.dtype(np.uint8) else 0.0
    for j in range(data.shape[1]):
        def rd(seg, a, b, _j=j):
            return np.array(data[a:b, _j])

        rec.signals.append(Signal(name=f"channel {j + 1}", unit="", rate=float(rate),
                                  segments=[(0.0, int(data.shape[0]))], read=rd, dtype=dt, conversion=conv,
                                  offset=off, description="WAV channel (no name or unit in the file)"))
    return [rec]


def read_acq(path: Path, fmt: str, ctx) -> list:
    import bioread

    try:
        f = bioread.read_file(str(path))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"bioread could not read the file: {type(e).__name__}: {e}") from None
    rec = SourceRecording(path=path, source_format="acq")
    try:
        st = f.earliest_marker_created_at
        if st is not None:
            rec.start_time = st
    except Exception:  # noqa: BLE001
        pass
    for ch in f.channels:
        x = np.asarray(ch.data)
        if x.size == 0:
            continue

        def rd(seg, a, b, _x=x):
            return np.asarray(_x[a:b])

        rec.signals.append(Signal(name=str(ch.name).strip(), unit=str(ch.units or "").strip(),
                                  rate=float(ch.samples_per_second), segments=[(0.0, int(x.size))], read=rd,
                                  dtype=x.dtype))
    base = float(f.samples_per_second)
    for m in getattr(f, "event_markers", []) or []:
        rec.markers.append(Marker(float(m.sample_index) / base, None, str(m.text or "").strip() or "marker",
                                  "marker", {"channel": m.channel_number} if m.channel_number is not None else {}))
    if not rec.signals:
        raise Skip("no channels with data")
    return [rec]
