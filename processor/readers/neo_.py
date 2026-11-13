"""
Formats read with Neo's raw IO layer (NeuralEnsemble, BSD-3-Clause): Axon ABF (ABF1/ABF2, gap-free
and episodic), Spike2 .smr (CED 32-bit, pure Python, no sonpy), and Intan, Blackrock, Plexon,
BrainVision, Open Ephys (binary) and TDT as Neo reads them. Stored integers are kept with Neo's
per-channel gain and offset. Each Neo segment (an ABF sweep, a Spike2 recording block) becomes a
segment of the recording at its own start time; event and epoch channels become markers.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from processor.model import Marker, Signal, SourceRecording, Skip

IO = {"abf": "AxonRawIO", "smr": "Spike2RawIO", "intan": "IntanRawIO", "blackrock": "BlackrockRawIO",
      "plexon": "PlexonRawIO", "brainvision": "BrainVisionRawIO", "openephys": "OpenEphysBinaryRawIO",
      "tdt": "TdtRawIO"}
FOLDER = {"openephys", "tdt"}


def _open(path: Path, fmt: str):
    import neo.rawio as nr

    cls = getattr(nr, IO[fmt])
    if fmt in FOLDER:
        r = cls(dirname=str(path))
    else:
        r = cls(filename=str(path))
    r.parse_header()
    return r


def _rec_datetime(r) -> datetime | None:
    try:
        ann = r.raw_annotations["blocks"][0]
        for d in (ann, ann["segments"][0]):
            v = d.get("rec_datetime")
            if isinstance(v, datetime):
                return v
    except Exception:  # noqa: BLE001
        pass
    return None


def read(path: Path, fmt: str, ctx) -> list:
    try:
        r = _open(path, fmt)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Neo ({IO[fmt]}) could not read the file: {type(e).__name__}: {e}") from None
    rec = SourceRecording(path=path, source_format=fmt)
    rec.start_time = _rec_datetime(r)
    hdr = r.header
    streams = hdr["signal_streams"]
    chans = hdr["signal_channels"]
    n_blocks = r.block_count()
    if n_blocks > 1:
        rec.notes.append(f"{n_blocks} Neo blocks: only the first is read")
    b = 0
    n_seg = r.segment_count(b)
    seg_t0 = [float(r.segment_t_start(b, s)) for s in range(n_seg)]
    t_ref = min(seg_t0) if seg_t0 else 0.0
    for si in range(len(streams)):
        sid = streams["id"][si]
        idx = np.flatnonzero(chans["stream_id"] == sid)
        for local, ci in enumerate(idx):
            c = chans[ci]
            fs = float(c["sampling_rate"])
            segs, seg_idx = [], []
            for s in range(n_seg):
                try:
                    n = int(r.get_signal_size(b, s, si))
                    t0 = float(r.get_signal_t_start(b, s, si))
                except Exception:  # noqa: BLE001
                    continue
                if n > 0:
                    segs.append((t0 - t_ref, n))
                    seg_idx.append(s)
            if not segs:
                continue

            def rd(seg, a, bb, _si=si, _local=local, _segs=seg_idx):
                x = r.get_analogsignal_chunk(block_index=0, seg_index=_segs[seg], i_start=a, i_stop=bb,
                                             stream_index=_si, channel_indexes=[_local])
                return np.asarray(x[:, 0])

            dtype = np.dtype(c["dtype"])
            rec.signals.append(Signal(name=str(c["name"]).strip(), unit=str(c["units"]).strip(), rate=fs,
                                      segments=segs, read=rd, dtype=dtype, conversion=float(c["gain"]),
                                      offset=float(c["offset"])))
    # events and epochs (Spike2 markers and keyboard/text marks, ABF tags, trigger event channels)
    evs = hdr["event_channels"]
    for ei in range(len(evs)):
        ename = str(evs["name"][ei]).strip()
        for s in range(n_seg):
            try:
                ts, du, lab = r.get_event_timestamps(block_index=b, seg_index=s, event_channel_index=ei)
                t = r.rescale_event_timestamp(ts, dtype="float64", event_channel_index=ei)
                d = r.rescale_epoch_duration(du, dtype="float64", event_channel_index=ei) if du is not None else None
            except Exception:  # noqa: BLE001
                continue
            for k in range(len(t)):
                text = str(lab[k]).strip() if lab is not None and len(lab) > k else ""
                label = f"{ename}: {text}" if text and text != ename else ename
                rec.markers.append(Marker(float(t[k]) - t_ref, float(d[k]) if d is not None else None, label,
                                          "marker", {"channel_name": ename}))
    if not rec.signals:
        raise Skip("no continuous signals in the file (events or spikes only)")
    return [rec]
