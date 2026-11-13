"""
Channel typing, units and names (conventions.md, "channels.csv and channel types").

Order: a user channels-map.csv (first matching row wins), then name heuristics (on the channel's
own name first, then on the full source path, e.g. HDF5 'neural/stim_timing' is typed by
'stim_timing'), then a content check (a channel of two levels, 0/1 steps, is `stim`), else
`other`. Every decision is written to channels.csv `typing`.

Units: ECG, nerve and EMG are stored in volts, blood pressure in mmHg, heart rate in bpm,
temperature in degC; the factor from the source unit is `scale`. A channel whose source gives no
physical unit (or a unit that does not fit its type, e.g. BP in volts) is `calibrated=false` and
keeps the source values (`a.u.` when there is no unit at all).
"""
from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

TYPES = ("ecg", "bp", "resp", "ppg", "nerve", "emg", "stim", "hr", "temp", "pressure", "other")

# Token boundaries: a short name like ECG, BP or HR must not be a letter run inside a longer word
# ('heart' must not match ART, 'capacity' must not match CAP). Digits and punctuation are boundaries.
_B, _E = r"(?<![a-z])", r"(?![a-z])"


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(f"{_B}(?:{a}){_E}" for a in alts), re.IGNORECASE)


# Checked in this order on whole tokens; the first that matches wins (conventions.md, "channels.csv and channel
# types").
HEURISTICS: list[tuple[str, re.Pattern]] = [
    # pressures that are not arterial first: venous, ventricular, atrial, pulmonary, organ pressures
    ("pressure", _rx(r"(?:central[ _-]?)?venous(?:[ _-]?pressure)?", r"(?:left|right)?[ _-]?ventricular(?:[ _-]?pressure)?",
                     r"(?:left|right)?[ _-]?atrial(?:[ _-]?pressure)?", r"pulmonary(?:[ _-]?\w+)*[ _-]?pressure",
                     r"cvp", r"lvp", r"rvp", r"pap", r"lap", r"rap", r"jvp", r"pcwp", r"lvedp",
                     r"(?:bladder|vesical|intravesical|detrusor|gastric|airway|tracheal|esophageal|oesophageal|"
                     r"rectal|abdominal|intra-?abdominal|urethral|colonic|intracranial|pleural)[ _-]?(?:pressure|press|p)",
                     r"icp", r"cmh2o", r"pves", r"pdet", r"pabd")),
    # heart rate before ECG: "HR (ECG)" is a rate derived from the ECG, not an ECG
    ("hr", _rx(r"hr", r"heart[ _-]?rate", r"bpm", r"pulse[ _-]?rate")),
    ("ecg", _rx(r"ecg\d*", r"ekg\d*", r"lead[ _-]?(?:i{1,3}|avr|avl|avf|v[1-6])")),
    ("bp", _rx(r"bp", r"abp", r"art", r"nibp", r"finapres", r"map", r"sbp", r"dbp", r"fap", r"bpw",
               r"blood[ _-]?pressure", r"arterial[ _-]?(?:blood[ _-]?)?pressure", r"aortic[ _-]?pressure")),
    # an unspecified pressure ('Pressure 1') is not assumed arterial
    ("pressure", _rx(r"pressure", r"press")),
    ("resp", _rx(r"resp", r"respiration", r"respiratory", r"breath", r"breaths", r"breathing", r"thorax", r"thoracic",
                 r"abdomen", r"airflow", r"pneumo(?:tach|tachograph|graph)?", r"plethysmograph(?:y)?", r"spirometer",
                 r"spirometry", r"rip", r"co2", r"etco2", r"thermistor")),
    # EMG before stimulation: "EMG during stimulation" is an EMG
    ("emg", _rx(r"emg\d*", r"muscle")),
    # stimulation before pulse: "stim pulse" and "pulse train" are stimulation, a bare "pulse" is a PPG
    ("stim", _rx(r"stim\w*", r"trig(?:ger)?s?", r"ttl", r"marker", r"pulse[ _-]?train", r"stimamp", r"ctrlamp",
                 r"sync")),
    ("ppg", _rx(r"ppg", r"pleth", r"photopleth\w*", r"pulse", r"pulse[ _-]?ox(?:imeter|imetry)?", r"finger[ _-]?pulse",
                r"spo2[ _-]?pleth")),
    ("nerve", _rx(r"eng", r"nerve", r"vagus", r"vagal", r"cuff", r"cap", r"cnap", r"ecap", r"neural", r"neurogram",
                  r"msna", r"sna", r"rsna", r"ssna")),
    ("temp", _rx(r"temp(?:erature)?", r"thermo(?:meter|couple)?")),
]
# A processed copy of another channel keeps its type and gets channels.csv derived_from
DERIVED = _rx(r"processed", r"filtered", r"filt", r"clean", r"cleaned", r"smoothed", r"rectified", r"integrated",
              r"envelope", r"derived", r"differentiated")


def is_derived(original: str, leaf: str = "") -> bool:
    return bool(DERIVED.search(leaf or original) or DERIVED.search(original))


# NWB unit per type and how to reach it from the units sources use
VOLT = {"v": 1.0, "volt": 1.0, "volts": 1.0, "mv": 1e-3, "millivolt": 1e-3, "millivolts": 1e-3,
        "uv": 1e-6, "µv": 1e-6, "μv": 1e-6, "microvolt": 1e-6, "microvolts": 1e-6, "nv": 1e-9, "kv": 1e3}
MMHG = {"mmhg": 1.0, "mm hg": 1.0, "torr": 1.0, "kpa": 7.500617, "pa": 0.007500617, "cmh2o": 0.7355592,
        "cm h2o": 0.7355592, "mbar": 0.7500617}
BPM = {"bpm": 1.0, "beats/min": 1.0, "beats per minute": 1.0, "1/min": 1.0, "/min": 1.0, "min-1": 1.0,
       "hz": 60.0, "1/s": 60.0}
DEGC = {"degc": 1.0, "°c": 1.0, "c": 1.0, "deg c": 1.0, "celsius": 1.0}
TARGET = {"ecg": ("V", VOLT), "nerve": ("V", VOLT), "emg": ("V", VOLT), "bp": ("mmHg", MMHG),
          "hr": ("bpm", BPM), "temp": ("degC", DEGC)}


def norm_unit(u: str) -> str:
    s = (u or "").strip()
    s = s.replace("μ", "u").replace("µ", "u")
    return s


def unit_key(u: str) -> str:
    return re.sub(r"\s+", " ", norm_unit(u).lower()).strip()


@dataclass
class MapRow:
    pattern: re.Pattern
    type: str
    unit: str
    scale: float | None
    site: str
    name: str
    path_glob: str


def read_channels_map(path: Path) -> list[MapRow]:
    """channels-map.csv: pattern (case-insensitive regex on the original name), type, unit, scale, site,
    name; optional path_glob (only files whose path matches)."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, r in enumerate(csv.DictReader(fh), 2):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            pat = r.get("pattern", "")
            if not pat:
                continue
            t = r.get("type", "").lower()
            if t and t not in TYPES:
                raise ValueError(f"{path.name} line {i}: type {t!r} is not one of {', '.join(TYPES)}")
            try:
                rx = re.compile(pat, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"{path.name} line {i}: pattern {pat!r}: {e}") from None
            sc = r.get("scale", "")
            try:
                scale = float(sc) if sc else None
            except ValueError:
                raise ValueError(f"{path.name} line {i}: scale {sc!r} is not a number") from None
            name = r.get("name", "")
            if name and not re.fullmatch(r"[a-z0-9-]+", name):
                raise ValueError(f"{path.name} line {i}: name {name!r} must be lowercase letters, digits and -")
            rows.append(MapRow(rx, t, r.get("unit", ""), scale, r.get("site", ""), name, r.get("path_glob", "")))
    return rows


def map_match(rows: list[MapRow], original: str, leaf: str, source_rel: str) -> MapRow | None:
    for r in rows:
        if r.path_glob and not (fnmatch.fnmatch(source_rel, r.path_glob)
                                or fnmatch.fnmatch(Path(source_rel).name, r.path_glob)):
            continue
        if r.pattern.search(original) or (leaf and r.pattern.search(leaf)):
            return r
    return None


def type_by_name(original: str, leaf: str = "") -> str | None:
    for text in ([leaf] if leaf and leaf != original else []) + [original]:
        for t, rx in HEURISTICS:
            if rx.search(text):
                return t
    return None


def type_by_content(values) -> str | None:
    """Two levels with steps between them (a TTL or gate line) is a stimulation/trigger channel."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 100:
        return None
    u = np.unique(v[: min(v.size, 200_000)])
    if 2 <= u.size <= 3 and np.count_nonzero(np.diff(v) != 0) >= 2:
        return "stim"
    return None


@dataclass
class Calibration:
    unit: str          # NWB unit
    scale: float       # source unit -> NWB unit
    calibrated: bool
    note: str


def calibrate(ch_type: str, source_unit: str, user_unit: str = "", user_scale: float | None = None) -> Calibration:
    """NWB unit and factor for a channel of type `ch_type` whose source unit is `source_unit`. A channels-map
    unit/scale says: source value * scale = value in user_unit."""
    scale0 = 1.0
    unit = norm_unit(source_unit)
    note = ""
    if user_unit or user_scale is not None:
        scale0 = user_scale if user_scale is not None else 1.0
        unit = norm_unit(user_unit) or unit
        note = "unit/scale from channels-map.csv"
    key = unit_key(unit)
    if ch_type in TARGET:
        target, table = TARGET[ch_type]
        if key in table:
            return Calibration(target, scale0 * table[key], True, note)
        if ch_type == "hr" and not key:
            return Calibration("bpm", scale0, True, "no unit in the source: bpm assumed for a heart-rate channel")
        if key in ("", "a.u.", "au", "arb", "arbitrary", "n/a", "none", "-"):
            return Calibration("a.u.", scale0, False, note or "no physical unit in the source")
        return Calibration(unit, scale0, False,
                           (note + "; " if note else "") + f"unit {unit!r} is not a {target} unit: values kept as recorded")
    if ch_type == "pressure":
        if key in ("cmh2o", "cm h2o"):
            return Calibration("cmH2O", scale0, True, note)
        if key in MMHG:
            return Calibration("mmHg", scale0 * MMHG[key], True, note)
        if key in VOLT:  # a transducer's output voltage, not a pressure
            return Calibration("V", scale0 * VOLT[key], False,
                               (note + "; " if note else "") + "pressure transducer output in volts: not calibrated")
    if not key or key in ("a.u.", "au", "arb", "none", "-", "n/a"):
        return Calibration("a.u.", scale0, False, note or "no physical unit in the source")
    if key in VOLT:  # a voltage on a channel of another type: volts, spelled V
        return Calibration("V", scale0 * VOLT[key], True, note)
    return Calibration(unit, scale0, True, note)


def nwb_names(types: list[str], user_names: list[str]) -> list[str]:
    """`<type>` for the first channel of a type, `<type>-2`, `<type>-3`, ... for the next; a channels-map
    name wins (made unique with -2, -3 when it repeats)."""
    out, used, count = [], set(), {}
    for t, un in zip(types, user_names):
        if un:
            base = un
        else:
            count[t] = count.get(t, 0) + 1
            base = t if count[t] == 1 else f"{t}-{count[t]}"
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}-{k}"
        used.add(name)
        out.append(name)
    return out
