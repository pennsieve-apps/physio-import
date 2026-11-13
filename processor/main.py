"""
physio-import: read physiological recordings in common formats and write one NWB file per
recording plus index tables (conventions.md of the physiology workflows).

  INPUT_DIR   recordings anywhere below (EDF/BDF, WFDB, ABF, Spike2 .smr, BIOPAC .acq, NWB, HDF5,
              MATLAB .mat, CSV/TSV/TXT/XLSX, WAV, and what Neo reads), plus optional
              channels-map.csv, recordings-map.csv, layout.yaml, SDS subjects.xlsx / performances.xlsx
  OUTPUT_DIR/physio/   recordings.csv  channels.csv  events.tsv  provenance.json
                       sub-<subject>/[ses-<session>/]<recording_id>.nwb
  OUTPUT_DIR/qc/physio-import/   qc.csv  nwbinspector.csv  figures/<recording_id>_overview.png
                                 summary.json  provenance.json

One recording at a time, each channel read a slice at a time; a recording that fails is logged
and skipped. Exit codes: 0 ok, 1 no usable input / every recording failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: HOME and TMPDIR may be fresh, empty folders and everything else read-only.
    Writable caches go under TMPDIR; nothing is written to the image."""
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    for var, default in (("TMPDIR", tmp), ("HOME", tmp / "home")):
        p = Path(os.environ.get(var) or default)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ[var] = str(p)
    mpl = Path(os.environ["TMPDIR"]) / "mpl"
    baked = Path(os.environ.get("PHYSIO_IMPORT_MPL_CACHE", "/opt/physio-import/matplotlib"))
    if not os.environ.get("MPLCONFIGDIR") or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
        mpl.mkdir(parents=True, exist_ok=True)
        if baked.is_dir():
            for f in baked.glob("fontlist-*.json"):
                target = mpl / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


_prepare_environment()

from processor import resources as R  # noqa: E402  (no numpy import)

_requested = os.environ.get("THREADS", "0").strip() or "0"
try:
    _THREADS, _THREADS_WHY = R.resolve_workers(int(_requested), per_worker_mb=300, reserve_mb=600)
except ValueError:
    _THREADS, _THREADS_WHY = 1, f"THREADS={_requested!r} is not an integer"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import channels as C  # noqa: E402
from processor import discover as D  # noqa: E402
from processor import metadata as M  # noqa: E402
from processor import nwb_out as N  # noqa: E402
from processor import qc as Q  # noqa: E402
from processor import readers  # noqa: E402
from processor.model import Skip  # noqa: E402
from processor.readers import layout as L  # noqa: E402
from processor.readers.common import Ctx  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("physio-import")
for _noisy in ("matplotlib", "hdmf", "pynwb", "neo", "h5py", "nwbinspector"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

APP_NAME = "physio-import"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
READ_CHUNK = 1_048_576
CITATIONS = ["10.7554/eLife.78362", "10.3389/fninf.2014.00010", "10.13026/9njx-6322",
             "10.1161/01.CIR.101.23.e215"]
DEFAULTS = {"SPECIES": "auto", "TIME_COLUMN": "", "SAMPLING_RATE": "0", "EXCLUDE_FOLDERS": "derivative,derivatives,code,docs",
            "FIGURES": "true", "INSPECTOR": "true", "THREADS": "0"}
REC_COLS = ["recording_id", "subject", "session", "label", "path", "species", "start_time", "duration_s", "recorded_s",
            "n_segments", "channel_types", "source_format", "source_path", "metadata_source", "age", "sex",
            "task", "run"]  # task, run: BIDS entities of the run (empty for other sources)
CH_COLS = ["recording_id", "channel", "type", "original_name", "unit", "sampling_rate_hz", "n_samples", "calibrated",
           "scale", "site", "typing", "status", "status_description"]
EV_COLS = ["recording_id", "onset", "duration", "trial_type", "event_id", "amplitude", "amplitude_unit",
           "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s", "n_pulses", "polarity",
           "waveform", "site", "contact", "condition", "label", "source"]
QC_COLS = ["recording_id", "channel", "check", "value", "note"]
INSP_COLS = ["recording_id", "importance", "check", "object", "message"]
# NWB Inspector rates missing subject metadata CRITICAL (a DANDI upload requirement). A source file that has no
# age or sex cannot be given one, so these are reported (nwbinspector.csv, summary warnings) but do not fail the
# recording; every other CRITICAL issue (schema, data shape, timing) does.
METADATA_CHECKS = {"check_subject_exists", "check_subject_age", "check_subject_id_exists", "check_subject_sex",
                   "check_subject_weight"}
RESEARCH = "Research output: an import check, not a clinical assessment of the signals."


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("pynwb", "hdmf", "nwbinspector", "h5py", "neo", "wfdb", "pyedflib", "pymatreader", "bioread", "pandas",
                "openpyxl", "numpy", "scipy", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def read_params() -> tuple[dict, dict]:
    raw = {k: os.environ.get(k, v).strip() for k, v in DEFAULTS.items()}
    p = {}
    sp = raw["SPECIES"].lower() or "auto"
    if sp not in ("auto", *M.SPECIES):
        raise ValueError(f"SPECIES must be auto or one of {', '.join(M.SPECIES)}, got {raw['SPECIES']!r}")
    p["species"] = "" if sp == "auto" else sp
    p["time_column"] = raw["TIME_COLUMN"]
    try:
        p["sampling_rate"] = float(raw["SAMPLING_RATE"] or 0)
    except ValueError:
        raise ValueError(f"SAMPLING_RATE must be a number, got {raw['SAMPLING_RATE']!r}") from None
    if p["sampling_rate"] < 0 or not math.isfinite(p["sampling_rate"]):
        raise ValueError("SAMPLING_RATE must be 0 (from the file) or a positive number")
    p["exclude"] = {x.strip() for x in raw["EXCLUDE_FOLDERS"].split(",") if x.strip()}
    for k in ("FIGURES", "INSPECTOR"):
        v = raw[k].lower()
        if v not in TRUE + FALSE:
            raise ValueError(f"{k} must be true or false, got {raw[k]!r}")
        p[k.lower()] = v in TRUE
    try:
        int(raw["THREADS"] or 0)
    except ValueError:
        raise ValueError(f"THREADS must be an integer, got {raw['THREADS']!r}") from None
    return raw, p


def fmt(v, nd: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.{nd}g}"
    return str(v)


def peak_rss_mb() -> float:
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(Path(f).read_text().strip()) / 2**20, 1)
        except (OSError, ValueError):
            continue
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)


class Progress:
    def __init__(self, what: str):
        self.what, self.t = what, time.monotonic()

    def tick(self, text: str) -> None:
        if time.monotonic() - self.t >= 60:
            log.info("%s: %s", self.what, text)
            self.t = time.monotonic()


# --------------------------------------------------------------------------------- one recording
def ids_for(rec, rel: str, rmap, sds: M.SdsInfo, params: dict, n_from_file: int) -> dict:
    parts = Path(rel).parts
    fo = M.folder_ids(list(parts[:-1]), Path(rel).stem)
    entry = M.map_lookup(rmap, rel)
    meta_kind = "sds" if any(x.endswith(".xlsx") for x in sds.sources) else "bids"
    sources = []
    subject = session = label = species = ""
    if entry:
        subject, session, label, species = entry.subject, entry.session, entry.label, entry.species
        sources.append("mapping")
    if not subject and fo.get("sub"):
        subject = fo["sub"]
        sources.append(meta_kind)
    if not subject and rec.subject_id:
        subject = rec.subject_id
        sources.append("file")
    stem = Path(rel).name
    for ext in (".smrx.csv", ".smr.csv", ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".hea", ".edf", ".bdf", ".abf",
                ".smr", ".acq", ".nwb", ".h5", ".hdf5", ".hdf", ".mat", ".wav"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    if not subject:
        subject = M.clean_label(stem) or "unknown"
    perf = fo.get("perf", "")
    if not session:
        if fo.get("ses"):
            session = fo["ses"]
        elif perf:
            session = M.session_from_perf(perf, subject)
        elif rec.session_id:
            session = rec.session_id
    if not label:
        base = stem + (f"-{rec.key}" if rec.key and n_from_file > 1 else "")
        label = rec.label_hint or M.label_from_stem(base, subject, session, perf, rec.source_format)
    subj_key = M.clean_label(subject)
    sinfo = M.subject_info(sds, rec.path, subj_key)
    if not species and sinfo.get("species"):
        species = sinfo["species"]
        if meta_kind not in sources:
            sources.append(meta_kind)
    if not species and rec.species:
        species = rec.species
        if "file" not in sources:
            sources.append("file")
    species = species or params["species"]
    age = rec.age or sinfo.get("age", "")
    sex = rec.sex or sinfo.get("sex", "")
    if (rec.age or rec.sex) and "file" not in sources:
        sources.append("file")
    start = rec.start_time
    if start is None and perf and M.clean_label(perf).lower() in sds.performances:
        start = sds.performances[M.clean_label(perf).lower()]
        if "sds" not in sources:
            sources.append("sds")
    return {"subject": M.clean_label(subject), "session": M.clean_label(session), "label": M.clean_label(label),
            "species": species, "age": age, "sex": sex, "start": start,
            "metadata_source": "|".join(sources) or "none"}


def zoom_window(sigs) -> tuple[float, float]:
    segs = sorted({(s, n / sig.rate) for sig in sigs for s, n in sig.segments})
    for s, d in segs:
        if d >= Q.ZOOM_S:
            z0 = s + min(0.25 * d, d - Q.ZOOM_S)
            return z0, z0 + Q.ZOOM_S
    s0 = segs[0][0] if segs else 0.0
    return s0, s0 + Q.ZOOM_S


def chunks(sig):
    """(segment, a, b, t0 of the chunk) over the whole signal."""
    for s, (start, n) in enumerate(sig.segments):
        for a in range(0, n, READ_CHUNK):
            b = min(n, a + READ_CHUNK)
            yield s, a, b, start + a / sig.rate


def process(rec, rid: str, meta: dict, rel: str, cmap, out_physio: Path, figdir: Path, params: dict,
            qc_rows: list, insp_rows: list) -> dict:
    t_rec = time.monotonic()
    span = max(st + n / sig.rate for sig in rec.signals for st, n in sig.segments)
    zoom = zoom_window(rec.signals)
    prog = Progress(rid)
    # typing
    typed = []
    for sig in rec.signals:
        leaf = sig.leaf or sig.name
        row = C.map_match(cmap, sig.name, leaf, rel)
        if row and row.type:
            t, typing = row.type, "mapping"
        elif sig.type_hint:  # the format says (BIDS column names)
            t, typing = sig.type_hint, "name"
        else:
            t = C.type_by_name(sig.name, leaf)
            typing = "name" if t else ""
            if not t:
                s0, n0 = sig.segments[0]
                x = sig.read(0, 0, min(n0, 200_000)).astype(float) * sig.conversion + sig.offset
                t = C.type_by_content(x)
                typing = "content" if t else "default"
                t = t or "other"
        cal = C.calibrate(t, sig.unit, row.unit if row else "", row.scale if row else None)
        typed.append((sig, t, typing, cal, row))
    names = C.nwb_names([t for _, t, _, _, _ in typed], [(r.name if r else "") for _, _, _, _, r in typed])
    # pass 1: checks, envelope, storage decision
    plans, ch_rows = [], []
    for (sig, t, typing, cal, row), name in zip(typed, names):
        st = Q.ChannelStats(rate=sig.rate, span=span, zoom=zoom)
        is_float = sig.dtype.kind == "f"
        has_invalid = False
        rmin, rmax = math.inf, -math.inf
        for s, a, b, t0 in chunks(sig):
            raw = np.asarray(sig.read(s, a, b))
            times = None
            if sig.segment_times is not None:
                times = np.asarray(sig.segment_times(s, a, b), float)
            if sig.invalid is not None and raw.dtype.kind in "iu":
                bad = raw == sig.invalid
                has_invalid = has_invalid or bool(bad.any())
            else:
                bad = None
            x = raw.astype(np.float64) * sig.conversion + sig.offset
            if bad is not None and bad.any():
                x[bad] = np.nan
            st.add(x * cal.scale, t0, times, is_float)
            if is_float:
                st.check_quant(raw)
                if raw.size:
                    with np.errstate(invalid="ignore"):
                        rmin = min(rmin, float(np.nanmin(raw)) if np.isfinite(raw).any() else rmin)
                        rmax = max(rmax, float(np.nanmax(raw)) if np.isfinite(raw).any() else rmax)
            prog.tick(f"checking {name} ({sig.name}), segment {s + 1}/{len(sig.segments)}")
        conv, off = sig.conversion * cal.scale, sig.offset * cal.scale
        storage = "source integers"
        if sig.dtype.kind in "iu" and not has_invalid:
            plan = N.Plan(name, "", cal.unit, conv, off, sig.dtype, lambda r: r, sig)
        elif sig.dtype.kind in "iu":
            inv, c0, o0 = sig.invalid, sig.conversion, sig.offset
            plan = N.Plan(name, "", cal.unit, cal.scale, 0.0, np.dtype(np.float32),
                          lambda r, _i=inv, _c=c0, _o=o0: np.where(r == _i, np.nan, r * _c + _o).astype(np.float32),
                          sig)
            storage = "float32 (the source marks missing samples, kept as NaN)"
        else:
            idt = st.int_dtype(rmin, rmax) if math.isfinite(rmin) else None
            if idt is not None:
                q = st.quant
                plan = N.Plan(name, "", cal.unit, conv * q, off, idt,
                              lambda r, _q=q, _d=idt: np.round(np.asarray(r, float) / _q).astype(_d), sig)
                storage = f"{idt.name} steps of {q:g} (lossless: every source value is a multiple of it)"
            elif max(abs(rmin), abs(rmax)) < 3e38 if math.isfinite(rmin) else True:
                plan = N.Plan(name, "", cal.unit, conv, off, np.dtype(np.float32),
                              lambda r: np.asarray(r, np.float32), sig)
                storage = "float32"
            else:  # values beyond float32's range
                plan = N.Plan(name, "", cal.unit, conv, off, np.dtype(np.float64),
                              lambda r: np.asarray(r, np.float64), sig)
                storage = "float64"
        status, sdesc, vals = st.status(t)
        notes = [n for n in ((sig.type_note if typing == "name" else ""), cal.note, sdesc) if n]
        plan.description = (f"{sig.name}: type {t} (by {typing}); "
                            f"{'calibrated' if cal.calibrated else 'not calibrated'}; stored as {storage}"
                            + (f"; {sig.description}" if sig.description else ""))
        plan.comments = f"status {status}" + (f": {sdesc}" if sdesc else "")
        plans.append(plan)
        site = (row.site if row and row.site else sig.site)
        ch_rows.append({"recording_id": rid, "channel": name, "type": t, "original_name": sig.name, "unit": cal.unit,
                        "sampling_rate_hz": fmt(sig.rate, 10), "n_samples": sig.n_samples,
                        "calibrated": "true" if cal.calibrated else "false", "scale": fmt(cal.scale, 10),
                        "site": site, "typing": typing, "status": status,
                        "status_description": "; ".join(notes), "_stats": st, "_storage": storage})
        for k in ("n_samples", "nan_fraction", "min", "max", "mean", "sd", "flat_fraction", "saturated_fraction",
                  "lag1_autocorrelation"):
            qc_rows.append([rid, name, k, fmt(vals[k]), f"{cal.unit}" if k in ("min", "max", "mean", "sd") else ""])
        qc_rows.append([rid, name, "status", status, sdesc])
        if status == "saturated":
            from processor.model import Marker
            for a, b, v in st.clipped_intervals():
                rec.markers.append(Marker(a, b - a, f"{name} clipped at {v:g} {cal.unit}", "artifact",
                                          {"channel_name": name, "source": "qc"}))
        qc_rows.append([rid, name, "storage", storage, ""])
    # NWB
    sub = f"sub-{meta['subject']}" + (f"/ses-{meta['session']}" if meta["session"] else "")
    nwb_rel = f"{sub}/{rid}.nwb"
    nwb_path = out_physio / nwb_rel
    subject = {"subject_id": meta["subject"], "species": M.LATIN.get(meta["species"], ""), "sex": meta["sex"] or "U",
               "age": meta["age"]}
    desc = (f"Physiology recording imported by physio-import from {rec.source_format} file {Path(rel).name}"
            + (f" ({rec.key})" if rec.key else "") + ".")
    t_w = time.monotonic()
    N.write(nwb_path, rid, meta["start"], desc, subject, plans, rec.markers, rec.stim_table, Path(rel).name)
    t_write = time.monotonic() - t_w
    issues = N.inspect(nwb_path) if params["inspector"] else []
    for i in issues:
        insp_rows.append([rid, i["importance"], i["check"], i["object"], i["message"]])
    crit = [i for i in issues if i["importance"] == "CRITICAL" and i["check"] not in METADATA_CHECKS]
    qc_rows.append([rid, "", "nwbinspector_messages", str(len(issues)),
                    ", ".join(sorted({i["importance"] for i in issues}))])
    if crit:
        nwb_path.unlink(missing_ok=True)
        raise RuntimeError(f"NWB Inspector: {len(crit)} CRITICAL issue(s): {crit[0]['check']}: {crit[0]['message']}")
    # overview figure
    figs = []
    if params["figures"]:
        figdir.mkdir(parents=True, exist_ok=True)
        rows = [{"channel": r["channel"], "original": r["original_name"], "unit": r["unit"], "type": r["type"],
                 "typing": r["typing"], "status": r["status"], "status_description": r["status_description"],
                 "stats": r["_stats"]} for r in ch_rows]
        nseg = max(len(s.segments) for s in rec.signals)
        Q.overview_png(figdir / f"{rid}_overview.png", rid, rows, span, zoom,
                       f"; {nseg} segments (gaps between them are blank)" if nseg > 1 else "")
        bad = [r for r in ch_rows if r["status"] != "good"]
        figs.append({"path": f"figures/{rid}_overview.png", "caption": (
            f"{rid}: every channel imported, the whole recording on the left (the band is the minimum to maximum "
            f"within each pixel) and a 10 s zoom on the right (shaded on the left). Check that each channel looks "
            f"like the signal its type says (ECG beats, a pressure wave, breathing) and that the units are plausible. "
            + (f"Flagged: {', '.join(r['channel'] + ' ' + r['status'] for r in bad)} (title in colour). " if bad else
               "No channel was flagged flat, saturated or noisy. ") + RESEARCH)})
    durations = [st + n / sig.rate for sig in rec.signals for st, n in sig.segments]
    recorded = max(sum(n / sig.rate for _, n in sig.segments) for sig in rec.signals)
    types = []
    for r in ch_rows:
        if r["type"] not in types:
            types.append(r["type"])
    log.info("%s: %d channel(s) [%s], %.1f s (%.1f s recorded, %d segment(s)), %d marker(s)%s; written in %.1fs, "
             "%.1fs in total", rid, len(ch_rows), ", ".join(f"{r['channel']}={r['original_name']}" for r in ch_rows),
             max(durations), recorded, max(len(s.segments) for s in rec.signals), len(rec.markers),
             f", {len(rec.stim_table)} trial(s) of parameters" if rec.stim_table else "", t_write,
             time.monotonic() - t_rec)
    return {"rec_row": {"recording_id": rid, "subject": meta["subject"], "session": meta["session"],
                        "label": meta["label"], "path": nwb_rel, "species": meta["species"],
                        "start_time": _start_str(meta["start"]), "duration_s": fmt(max(durations), 9),
                        "recorded_s": fmt(recorded, 9), "n_segments": max(len(s.segments) for s in rec.signals),
                        "channel_types": "|".join(types), "source_format": rec.source_format, "source_path": rel
                        + (f"#{rec.key}" if rec.key else ""), "metadata_source": meta["metadata_source"],
                        "age": meta["age"], "sex": meta["sex"], "task": rec.entities.get("task", ""),
                        "run": rec.entities.get("run", "")},
            "ch_rows": ch_rows, "figures": figs, "issues": issues,
            "headline": {"n_channels": len(ch_rows), "duration_s": round(max(durations), 3),
                         "recorded_s": round(recorded, 3), "n_markers": len(rec.markers),
                         "n_trials": len(rec.stim_table) if rec.stim_table else 0,
                         "n_channels_flagged": sum(r["status"] != "good" for r in ch_rows),
                         "nwbinspector_messages": len(issues)}}


def _start_str(st) -> str:
    if st is None:
        return ""
    if st.tzinfo is not None:
        st = st.astimezone(timezone.utc).replace(tzinfo=None)
    return st.isoformat(timespec="seconds" if st.microsecond == 0 else "milliseconds")


def events_rows(rid: str, markers) -> list[dict]:
    out = []
    for i, m in enumerate(sorted(markers, key=lambda m: m.onset), 1):
        row = {c: "n/a" for c in EV_COLS}
        row.update(recording_id=rid, onset=fmt(m.onset, 10), duration=fmt(m.duration, 10) if m.duration else "n/a",
                   trial_type=m.trial_type,
                   event_id=f"{ {'annotation': 'ann', 'artifact': 'art'}.get(m.trial_type, 'mrk') }-{i:06d}",
                   label=m.label.replace("\t", " ").replace("\n", " ") or "n/a",
                   source=(m.extra or {}).get("source", "marker"))
        extra = m.extra or {}
        row["annotator"] = str(extra.get("annotator", "n/a"))
        row["marker_channel"] = str(extra.get("channel_name", extra.get("channel", extra.get("column", "n/a"))))
        out.append(row)
    return out


# --------------------------------------------------------------------------------- main
def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw_params, params = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 2
    out_physio = output_dir / "physio"
    out_qc = output_dir / "qc" / APP_NAME
    figdir = out_qc / "figures"
    out_physio.mkdir(parents=True, exist_ok=True)
    out_qc.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s threads=%d (%s)", APP_NAME, app_version(), input_dir, output_dir,
             _THREADS, _THREADS_WHY)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s does not exist", input_dir)
        return 1
    found = D.find(input_dir, params["exclude"])

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(input_dir))
        except ValueError:
            return str(p)

    try:
        cmap = [r for p in sorted(found.config.get("channels_map", [])) for r in C.read_channels_map(p)]
        rmap = [e for p in sorted(found.config.get("recordings_map", [])) for e in M.read_recordings_map(p)]
        layouts = L.load_layouts(sorted(found.config.get("layout", [])))
    except (ValueError, OSError) as e:
        log.error("configuration file: %s", e)
        return 2
    sds = M.read_sds(sorted(found.config.get("sds", [])), log)
    M.read_participants(sorted(found.config.get("participants", [])), sds, log)
    for k, v in found.config.items():
        log.info("config %s: %s", k, ", ".join(rel(p) for p in v))
    for a, b in found.duplicates:
        log.info("skipped %s: identical to %s", rel(a), rel(b))
    if found.ignored:
        log.info("not recordings (ignored): %s", ", ".join(f"{k} x{v}" for k, v in sorted(found.ignored.items())))
    if not found.sources:
        log.error("no recording found under %s", input_dir)
        _write_tables(out_physio, out_qc, [], [], [], [], [])
        return 1
    log.info("%d source file(s) to read", len(found.sources))
    ctx = Ctx(time_column=params["time_column"], sampling_rate=params["sampling_rate"], layouts=layouts, log=log,
              input_root=input_dir)

    rec_rows, ch_rows, ev_rows, qc_rows, insp_rows = [], [], [], [], []
    per_rec: dict[str, dict] = {}
    headlines, figures, warnings_out = {}, [], []
    used_ids: set[str] = set()
    n_ok = n_failed = 0
    for i, (path, fmt_) in enumerate(found.sources, 1):
        r = rel(path)
        t0 = time.monotonic()
        log.info("[%d/%d] %s (%s)", i, len(found.sources), r, fmt_)
        try:
            recs = readers.read(path, fmt_, ctx)
        except Skip as e:
            log.warning("%s: skipped: %s", r, e)
            per_rec[r] = {"status": "skipped", "reason": str(e), "source_format": fmt_}
            warnings_out.append(f"{r}: skipped: {e}")
            continue
        except Exception as e:  # noqa: BLE001
            log.error("%s: failed to read: %s", r, e)
            log.debug(traceback.format_exc())
            per_rec[r] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "source_format": fmt_}
            warnings_out.append(f"{r}: failed: {e}")
            n_failed += 1
            continue
        for rec in recs:
            meta = ids_for(rec, r, rmap, sds, params, len(recs))
            rid = M.recording_id(meta["subject"], meta["session"], meta["label"])
            base, k = rid, 1
            while rid in used_ids:
                k += 1
                rid = f"{base}-{k}"
            used_ids.add(rid)
            try:
                res = process(rec, rid, meta, r, cmap, out_physio, figdir, params, qc_rows, insp_rows)
            except Exception as e:  # noqa: BLE001 - a recording that fails is logged and skipped
                log.error("%s: failed: %s", rid, e)
                log.info(traceback.format_exc())
                per_rec[rid] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "source": r}
                warnings_out.append(f"{rid}: failed: {e}")
                n_failed += 1
                continue
            finally:
                try:
                    rec.close()
                except Exception:  # noqa: BLE001
                    pass
            rec_rows.append(res["rec_row"])
            ch_rows.extend(res["ch_rows"])
            ev_rows.extend(events_rows(rid, rec.markers))
            figures.extend(res["figures"])
            headlines[rid] = res["headline"]
            per_rec[rid] = {"status": "ok", "seconds": round(time.monotonic() - t0, 2), "species": meta["species"],
                            "source": r, "notes": rec.notes[:20]}
            for c in res["ch_rows"]:
                if c["status"] != "good":
                    warnings_out.append(f"{rid}: {c['channel']} ({c['original_name']}) {c['status']}: "
                                        f"{c['status_description']}")
            if not meta["species"]:
                warnings_out.append(f"{rid}: species unknown (set SPECIES, or give it in recordings-map.csv or SDS "
                                    "subjects.xlsx): later steps assume human")
            n_ok += 1
    for a, b in found.duplicates:
        per_rec[rel(a)] = {"status": "skipped", "reason": f"identical to {rel(b)}"}

    _write_tables(out_physio, out_qc, rec_rows, ch_rows, ev_rows, qc_rows, insp_rows)
    finished = datetime.now(timezone.utc)
    provenance = {
        "app": APP_NAME, "version": app_version(), "variant": None, "tools": tool_versions(),
        "parameters": raw_params,
        "inputs": sorted(rel(p) for p, _ in found.sources) + sorted(rel(p) for v in found.config.values() for p in v),
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2), "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": int(raw_params["THREADS"] or 0), "used": _THREADS, "why": _THREADS_WHY},
        "recordings": per_rec, "citations": CITATIONS, "licence_notice": None,
    }
    for p in (out_physio / "provenance.json", out_qc / "provenance.json"):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2, ensure_ascii=False, default=str)
    uncal = sorted({r["recording_id"] for r in ch_rows if r["calibrated"] == "false" and r["type"] in ("ecg", "bp")})
    summary = {
        "app": APP_NAME, "version": app_version(), "recordings": headlines, "figures": figures,
        "tables": [
            {"path": "../../physio/recordings.csv", "caption": "One row per recording: ids, species, start time, "
             "duration, channel types, source file and where the metadata came from."},
            {"path": "../../physio/channels.csv", "caption": "One row per channel: NWB name, type and how it was "
             "decided (typing), unit, rate, whether it is calibrated, the factor applied (scale), and status."},
            {"path": "../../physio/events.tsv", "caption": "Markers and annotations found in the source files "
             "(BIDS-style events; stimulation parameters are added by stim-events)."},
            {"path": "qc.csv", "caption": "Per-channel checks: missing samples, range, flat and saturated fractions, "
             "lag-1 autocorrelation, status and how the samples are stored; NWB Inspector message counts."},
            {"path": "nwbinspector.csv", "caption": "NWB Inspector messages for every NWB file written (CRITICAL "
             "issues fail the recording)."}],
        "warnings": warnings_out,
    }
    if uncal:
        summary["notice"] = (f"Uncalibrated ECG or blood pressure in {len(uncal)} recording(s): their amplitudes are "
                             "relative (timing and shape only).")
    with open(out_qc / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    log.info("%d recording(s) written, %d failed, %d file(s) skipped in %.1fs, peak memory %.0f MB",
             n_ok, n_failed, sum(1 for v in per_rec.values() if v.get("status") == "skipped"),
             time.monotonic() - t_start, provenance["peak_rss_mb"])
    if n_ok == 0:
        log.error("no recording could be imported")
        return 1
    return 0


def _write_tables(out_physio: Path, out_qc: Path, rec_rows, ch_rows, ev_rows, qc_rows, insp_rows) -> None:
    def write_csv(path, cols, rows, delim=","):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=delim, lineterminator="\n")
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols] if isinstance(r, dict) else r)

    write_csv(out_physio / "recordings.csv", REC_COLS, rec_rows)
    write_csv(out_physio / "channels.csv", CH_COLS, ch_rows)
    extra = [c for c in ("annotator", "marker_channel") if any(r.get(c, "n/a") != "n/a" for r in ev_rows)]
    write_csv(out_physio / "events.tsv", EV_COLS + extra, ev_rows, "\t")
    write_csv(out_qc / "qc.csv", QC_COLS, qc_rows)
    write_csv(out_qc / "nwbinspector.csv", INSP_COLS, insp_rows)


if __name__ == "__main__":
    sys.exit(main())
