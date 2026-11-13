"""Reader dispatch: format name (discover.EXT) -> reader returning SourceRecordings."""
from __future__ import annotations

from pathlib import Path

from processor.discover import DEFERRED
from processor.model import Skip


def read(path: Path, fmt: str, ctx) -> list:
    if fmt in DEFERRED:
        raise Skip(DEFERRED[fmt])
    if fmt in ("edf", "bdf"):
        from processor.readers import edf
        return edf.read(path, fmt, ctx)
    if fmt == "wfdb":
        from processor.readers import wfdb_
        return wfdb_.read(path, fmt, ctx)
    if fmt == "nwb":
        from processor.readers import nwb_in
        return nwb_in.read(path, fmt, ctx)
    if fmt == "hdf5":
        from processor.readers import hdf5_, nwb_in
        if hdf5_.is_nwb(path):
            return nwb_in.read(path, "nwb", ctx)
        return hdf5_.read(path, fmt, ctx)
    if fmt == "mat":
        from processor.readers import mat_
        return mat_.read(path, fmt, ctx)
    if fmt in ("csv", "tsv", "txt", "xlsx"):
        from processor.readers import table
        return table.read(path, fmt, ctx)
    if fmt == "wav":
        from processor.readers import wav_acq
        return wav_acq.read_wav(path, fmt, ctx)
    if fmt == "acq":
        from processor.readers import wav_acq
        return wav_acq.read_acq(path, fmt, ctx)
    from processor.readers import neo_
    if fmt in neo_.IO:
        return neo_.read(path, fmt, ctx)
    raise Skip(f"no reader for format {fmt!r}")
