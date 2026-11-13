"""
Subjects, sessions, labels, species: where each recording's metadata comes from.

Order (conventions.md, "Recording ids, subjects, sessions"): a user recordings-map.csv
(path_glob, subject, session, label, species), BIDS/SDS-style folder names (sub-*, ses-*, perf-*),
the file's own metadata (NWB subject, WFDB comments, an HDF5 subject group), else the file stem.
Species, sex and age also come from SDS subjects.xlsx (and the SPECIES parameter fills what is
still unknown). SDS performances.xlsx gives a start time when the file has none.
"""
from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SPECIES = ("human", "pig", "dog", "rabbit", "rat", "mouse", "other")
LATIN = {"human": "Homo sapiens", "pig": "Sus scrofa", "dog": "Canis familiaris", "rabbit": "Oryctolagus cuniculus",
         "rat": "Rattus norvegicus", "mouse": "Mus musculus"}
_SPECIES_WORDS = [
    ("human", r"homo sapiens|human|patient|participant|volunteer"),
    ("pig", r"sus scrofa|swine|pig|porcine|minipig|yucatan|gottingen|göttingen"),
    ("dog", r"canis (?:lupus )?familiaris|dog|canine|beagle"),
    ("rabbit", r"oryctolagus|rabbit|new zealand white"),
    ("rat", r"rattus|rat|sprague|wistar|long[- ]evans|fischer 344"),
    ("mouse", r"mus musculus|mouse|mice|c57bl"),
]


def species_code(text: str) -> str:
    s = (text or "").strip().lower()
    if not s:
        return ""
    if s in SPECIES:
        return s
    for code, rx in _SPECIES_WORDS:
        if re.search(rf"\b(?:{rx})\b", s):
            return code
    return "other"


def sex_code(text: str) -> str:
    s = (text or "").strip().lower()
    return {"f": "F", "female": "F", "w": "F", "m": "M", "male": "M", "u": "U", "unknown": "U", "o": "O",
            "other": "O"}.get(s, "")


def age_iso(text: str) -> str:
    """'4 months' -> P4M, '20 years' -> P20Y, '23' (years) -> P23Y, 'P3M' kept; '' when not parseable."""
    s = (text or "").strip()
    if not s:
        return ""
    if re.fullmatch(r"P(?:\d+(?:\.\d+)?[YMWD])+(?:T(?:\d+(?:\.\d+)?[HMS])+)?", s):
        return s
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(y|yr|yrs|year|years|m|mo|month|months|w|wk|weeks?|d|days?)?\.?", s, re.I)
    if not m:
        return ""
    num, u = m.group(1), (m.group(2) or "y").lower()
    return f"P{num}{ {'y': 'Y', 'm': 'M', 'w': 'W', 'd': 'D'}[u[0]] }"


def clean_label(text: str) -> str:
    """Letters, digits and '-' only (conventions: inside each id label)."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-")
    return s


@dataclass
class MapEntry:
    path_glob: str
    subject: str = ""
    session: str = ""
    label: str = ""
    species: str = ""


def read_recordings_map(path: Path) -> list[MapEntry]:
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, r in enumerate(csv.DictReader(fh), 2):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            g = r.get("path_glob", "")
            if not g:
                continue
            sp = r.get("species", "")
            if sp and species_code(sp) not in SPECIES:
                raise ValueError(f"{path.name} line {i}: species {sp!r}")
            out.append(MapEntry(g, r.get("subject", ""), r.get("session", ""), r.get("label", ""),
                                species_code(sp) if sp else ""))
    return out


def map_lookup(entries: list[MapEntry], rel: str) -> MapEntry | None:
    for e in entries:
        if fnmatch.fnmatch(rel, e.path_glob) or fnmatch.fnmatch(Path(rel).name, e.path_glob):
            return e
    return None


@dataclass
class SdsInfo:
    # dataset folder (where subjects.xlsx / participants.tsv is) -> subject -> {species, sex, age}: a table only
    # applies to recordings below its own folder (two datasets may both have a sub-01)
    subjects: dict = field(default_factory=dict)
    performances: dict = field(default_factory=dict)   # perf id (without perf-) -> start datetime
    sources: list = field(default_factory=list)


def _norm_cols(df):
    df.columns = [re.sub(r"[\s_]+", " ", str(c)).strip().lower() for c in df.columns]
    return df


def strip_prefix(s: str, *prefixes: str) -> str:
    s = str(s).strip()
    for p in prefixes:
        if s.lower().startswith(p):
            return s[len(p):]
    return s


def read_sds(files: list[Path], log) -> SdsInfo:
    """SDS subjects.xlsx (subject id, species, sex, age) and performances.xlsx (performance id, start datetime)."""
    import pandas as pd

    info = SdsInfo()
    for f in files:
        try:
            df = _norm_cols(pd.read_excel(f, sheet_name=0, dtype=str))
        except Exception as e:  # noqa: BLE001
            log.warning("SDS %s: not read (%s)", f.name, e)
            continue
        name = f.name.lower()
        if name.startswith("subjects") and "subject id" in df.columns:
            for _, r in df.iterrows():
                sid = strip_prefix(r.get("subject id") or "", "sub-")
                if not sid or sid == "nan":
                    continue
                g = lambda k: ("" if (v := r.get(k)) is None or str(v) == "nan" else str(v))  # noqa: E731
                info.subjects.setdefault(str(f.parent), {})[clean_label(sid)] = {"species": species_code(g("species")), "sex": sex_code(g("sex")),
                                                   "age": age_iso(g("age"))}
            info.sources.append(str(f.name))
        elif name.startswith("performances") and "performance id" in df.columns:
            for _, r in df.iterrows():
                pid = strip_prefix(r.get("performance id") or "", "perf-")
                st = r.get("start datetime")
                if pid and st and str(st) != "nan":
                    try:
                        info.performances[clean_label(pid).lower()] = datetime.fromisoformat(str(st).strip())
                    except ValueError:
                        pass
            info.sources.append(str(f.name))
    return info


def folder_ids(folders: list[str], stem: str = "") -> dict:
    """sub-*, ses-*, perf-* from folder names (the whole name after the prefix) and from BIDS-style entities in
    the file name (split at '_')."""
    out = {}
    for part in folders:
        m = re.match(r"^(sub|ses|perf)-(.+)$", part, re.I)
        if m and m.group(1).lower() not in out:
            out[m.group(1).lower()] = m.group(2)
    for ent in stem.split("_"):
        m = re.match(r"^(sub|ses|perf)-(.+)$", ent, re.I)
        if m and m.group(1).lower() not in out:
            out[m.group(1).lower()] = m.group(2)
    return out


def session_from_perf(perf: str, subject: str) -> str:
    """perf-1001-A of subject 1001 -> A; perf-Bimodal01-set01-trial01-gain02 -> set01-trial01-gain02."""
    s = perf
    if subject and s.lower().startswith(subject.lower()):
        s = s[len(subject):].lstrip("-_")
    return s or perf


def label_from_stem(stem: str, subject: str, session: str, perf: str, fmt: str) -> str:
    """The file stem without the subject/session/performance it repeats; the format when nothing is left."""
    s = clean_label(stem)
    low = s.lower()
    for p in [f"perf-{perf}", f"sub-{subject}", f"subject-{subject}", f"subject{subject}", f"ses-{session}",
              subject, session]:
        p = clean_label(p).lower()
        if p and low.startswith(p) and (len(low) == len(p) or low[len(p)] == "-"):
            s = s[len(p):].lstrip("-")
            low = s.lower()
    if not s or s.lower() == clean_label(session).lower():
        return clean_label(fmt)
    return s


def recording_id(subject: str, session: str, label: str) -> str:
    rid = f"sub-{clean_label(subject)}"
    if session:
        rid += f"_ses-{clean_label(session)}"
    return rid + f"_rec-{clean_label(label)}"


def read_participants(files: list[Path], info: SdsInfo, log) -> None:
    """BIDS participants.tsv: participant_id with species, sex (or gender) and age (years) when present."""
    import pandas as pd

    for f in files:
        try:
            df = pd.read_csv(f, sep="\t", dtype=str, na_values=["n/a"])
        except Exception as e:  # noqa: BLE001
            log.warning("%s: not read (%s)", f.name, e)
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "participant_id" not in df.columns:
            continue
        for _, r in df.iterrows():
            sid = clean_label(strip_prefix(r["participant_id"] or "", "sub-"))
            if not sid:
                continue
            g = lambda k: "" if k not in r or pd.isna(r[k]) else str(r[k])  # noqa: E731
            cur = info.subjects.setdefault(str(f.parent), {}).setdefault(sid, {"species": "", "sex": "", "age": ""})
            cur["species"] = cur["species"] or species_code(g("species"))
            cur["sex"] = cur["sex"] or sex_code(g("sex") or g("gender"))
            cur["age"] = cur["age"] or age_iso(g("age"))
        info.sources.append(f.name)


def subject_info(info: SdsInfo, source: Path, subject: str) -> dict:
    """The subject's row from the deepest subjects.xlsx / participants.tsv whose folder contains the source."""
    src = str(source)
    for root in sorted(info.subjects, key=len, reverse=True):
        if src.startswith(root.rstrip("/") + "/") and subject in info.subjects[root]:
            return info.subjects[root][subject]
    return {}
