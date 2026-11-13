"""
What every reader returns: a source recording as signals, markers and per-trial parameters.

A Signal is read a slice at a time (`read(segment, start, stop)`), in the source's own storage
type (integers stay integers), with `conversion` and `offset` that turn stored values into the
source unit: value = stored * conversion + offset. A recording that was acquired trial by trial
or with gaps has several segments; each segment is regularly sampled at `rate` from its own
start (seconds from the recording start), unless `segment_times` gives explicit sample times.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Signal:
    name: str                                   # the name in the source (column, label, dataset path)
    unit: str                                   # unit as the source states it ('' when it states none)
    rate: float                                 # samples per second within a segment
    segments: list[tuple[float, int]]           # (start in s from the recording start, n samples)
    read: Callable[[int, int, int], np.ndarray]  # (segment, start, stop) -> stored values
    dtype: np.dtype                             # dtype `read` returns
    conversion: float = 1.0                     # stored * conversion + offset = value in `unit`
    offset: float = 0.0
    invalid: float | int | None = None          # sentinel the source uses for "no sample" (WFDB -32768)
    leaf: str = ""                              # last part of the name, typed before the full path
    site: str = ""
    description: str = ""
    # explicit sample times, (segment, start, stop) -> s from the recording start (irregular sampling, or a
    # source clock that differs from the nominal rate)
    segment_times: Callable[[int, int, int], np.ndarray] | None = None
    type_hint: str = ""                         # a type the reader knows from the format (BIDS column names)
    type_note: str = ""

    @property
    def n_samples(self) -> int:
        return int(sum(n for _, n in self.segments))


@dataclass
class Marker:
    onset: float                 # s from the recording start
    duration: float | None
    label: str
    trial_type: str = "marker"   # marker | annotation
    extra: dict = field(default_factory=dict)


@dataclass
class SourceRecording:
    path: Path                   # the file (or folder) read
    source_format: str           # edf, bdf, wfdb, abf, smr, acq, nwb, hdf5, mat, csv, tsv, xlsx, txt, wav, ...
    key: str = ""                # part of the file this recording is (an HDF5 group, a spreadsheet sheet)
    signals: list[Signal] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    stim_table: list[dict] | None = None   # per-trial parameters as found: start_time, stop_time, ...
    start_time: datetime | None = None     # naive = local wall clock; aware = an instant (UTC from epoch)
    subject_id: str = ""
    session_id: str = ""
    species: str = ""
    sex: str = ""
    age: str = ""
    label_hint: str = ""
    entities: dict = field(default_factory=dict)  # BIDS entities of the run (task, acq, run, ...)         # a better recording label than the file stem, when the reader knows one
    notes: list[str] = field(default_factory=list)
    close: Callable[[], None] = lambda: None


class Skip(Exception):
    """The file is not a recording this app can read (deferred format, unknown layout): skipped, not failed."""
