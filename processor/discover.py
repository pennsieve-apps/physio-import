"""
Find what to read under INPUT_DIR: recordings (by extension and content), and the optional
configuration files (channels-map.csv, recordings-map.csv, layout.yaml, SDS subjects.xlsx /
performances.xlsx). Recursive, follows symlinks (INPUT_DIR entries are symlinks on the platform),
never by fixed paths. Files with identical content (e.g. 'x - Copy.wav') are read once.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

EXT = {
    ".edf": "edf", ".bdf": "bdf", ".hea": "wfdb", ".abf": "abf", ".smr": "smr", ".acq": "acq", ".nwb": "nwb",
    ".h5": "hdf5", ".hdf5": "hdf5", ".hdf": "hdf5", ".mat": "mat", ".csv": "csv", ".tsv": "tsv", ".txt": "txt",
    ".xlsx": "xlsx", ".xls": "xlsx", ".wav": "wav",
    # other formats Neo reads (passed to Neo as they are)
    ".rhd": "intan", ".rhs": "intan", ".ns1": "blackrock", ".ns2": "blackrock", ".ns3": "blackrock",
    ".ns4": "blackrock", ".ns5": "blackrock", ".ns6": "blackrock", ".plx": "plexon", ".vhdr": "brainvision",
    # deferred: no reader with a licence we can ship
    ".smrx": "smrx", ".adicht": "adicht", ".adidat": "adicht",
}
DEFERRED = {
    "smrx": "Spike2 .smrx (64-bit) is not read: the only .smrx reader, CED's sonpy, is closed-source. Export the "
            "recording from Spike2 as .smr (File > Save As, 32-bit Spike2 data file) or as text/CSV (File > Export As, "
            "one column per channel with a Time column in seconds) and run again.",
    "adicht": "LabChart .adicht is not read: the only Linux reader is GPL-licensed. Export from LabChart as a MATLAB "
              "file (File > Export > MATLAB) or as text (File > Export > Text File, with time and channel titles) "
              "and run again.",
}
SDS_METADATA = {"dataset_description", "subjects", "samples", "performances", "manifest", "submission",
                "code_description", "sites", "resources", "curation", "code_parameters"}
CONFIG_NAMES = {"channels-map.csv": "channels_map", "recordings-map.csv": "recordings_map",
                "layout.yaml": "layout", "layout.yml": "layout"}
# Never recordings: documents, pictures, sidecars, and the Spike2 resource file
IGNORE_EXT = {".pdf", ".docx", ".doc", ".pptx", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".svg", ".json",
              ".md", ".html", ".htm", ".py", ".m", ".r", ".ipynb", ".zip", ".gz", ".s2rx", ".xml", ".log", ".yaml",
              ".yml", ".dss", ".ecg", ".atr", ".qrs", ".dat", ".ini", ".cff"}


@dataclass
class Found:
    sources: list[tuple[Path, str]] = field(default_factory=list)     # (path, format), sorted
    config: dict[str, list[Path]] = field(default_factory=dict)       # channels_map / recordings_map / layout / sds
    duplicates: list[tuple[Path, Path]] = field(default_factory=list)  # (skipped, identical to)
    ignored: dict[str, int] = field(default_factory=dict)             # extension -> count


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find(input_dir: Path, exclude_folders: set[str]) -> Found:
    found = Found()
    seen_real: set[str] = set()
    files: list[Path] = []
    folder_sources: list[tuple[Path, str]] = []
    excl = {e.lower() for e in exclude_folders}
    for root, dirs, names in os.walk(input_dir, followlinks=True):
        real = os.path.realpath(root)
        if real in seen_real:
            dirs[:] = []
            continue
        seen_real.add(real)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX" and d.lower() not in excl)
        rootp = Path(root)
        if "structure.oebin" in names:
            folder_sources.append((rootp, "openephys"))
            dirs[:] = []
            continue
        if any(n.lower().endswith(".tsq") for n in names):
            folder_sources.append((rootp, "tdt"))
            dirs[:] = []
            continue
        for n in sorted(names):
            if n.startswith(".") or n.startswith("~$"):
                continue
            files.append(rootp / n)

    by_size: dict[int, list[Path]] = {}
    for p in files:
        low = p.name.lower()
        if low in CONFIG_NAMES:
            found.config.setdefault(CONFIG_NAMES[low], []).append(p)
            continue
        ext = p.suffix.lower()
        stem = p.stem.lower()
        if ext in (".xlsx", ".xls") and stem in SDS_METADATA:
            if stem in ("subjects", "performances", "samples"):
                found.config.setdefault("sds", []).append(p)
            continue
        fmt = EXT.get(ext)
        if fmt is None:
            if ext in IGNORE_EXT or not ext:
                key = ext or "(no extension)"
            else:
                key = ext
            found.ignored[key] = found.ignored.get(key, 0) + 1
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        by_size.setdefault(size, []).append(p)
        found.sources.append((p, fmt))

    # identical files (same size, same SHA-256): keep the first in path order
    dup_of: dict[Path, Path] = {}
    for size, group in by_size.items():
        if len(group) < 2 or size == 0:
            continue
        first: dict[str, Path] = {}
        for p in sorted(group, key=lambda q: (("copy" in q.name.lower()), str(q))):
            h = _sha(p)
            if h in first:
                dup_of[p] = first[h]
            else:
                first[h] = p
    found.duplicates = sorted(dup_of.items())
    found.sources = sorted([(p, f) for p, f in found.sources if p not in dup_of] + folder_sources,
                           key=lambda x: str(x[0]))
    return found
