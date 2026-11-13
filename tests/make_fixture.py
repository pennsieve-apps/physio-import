"""
Synthetic physiology recordings in several source formats, with known contents (truth.json).

    python -m tests.make_fixture <out>      writes <out>/recordings/ (the input port) and <out>/truth.json

Only numpy, scipy, pandas, h5py and openpyxl are needed (the review generator image has them): EDF+,
WFDB (signals and beat annotations) and LabChart-style MATLAB files are written byte by byte here, so
the app's readers are checked against an independent writer. ABF and NWB inputs are added when pyabf
or pynwb are installed (the test environment).

Signals: an ECG with known R-peak times, an arterial pressure wave with known systolic and diastolic
values, respiration at a known rate, and a stimulation channel with known trains whose planted effect
is a bradycardia that grows with amplitude (5 bpm per mA in humans, 20 bpm per mA in rats). One
human-rate recording (60 bpm, 250 Hz), one rodent-rate recording (350 bpm, 1000 Hz), one without
blood pressure, a trial-by-trial HDF5 recording whose windows overlap, flat and clipped channels for
the checks, a duplicate file, and deferred .smrx/.adicht files.
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 7


# ------------------------------------------------------------------------------------------- signals
def beat_times(duration: float, hr_bpm: float, trains: list[dict], bpm_per_ma: float, rng) -> np.ndarray:
    """R-peak times: a baseline rate with a little variability, slowed by bpm_per_ma x amplitude during trains."""
    t, out = 0.3, []
    while t < duration - 0.2:
        hr = hr_bpm
        for tr in trains:
            if tr["onset"] <= t < tr["onset"] + tr["duration"]:
                hr -= bpm_per_ma * tr["amplitude_ma"]
        out.append(t)
        t += 60.0 / hr * (1 + 0.02 * rng.standard_normal())
    return np.asarray(out)


def ecg(t: np.ndarray, beats: np.ndarray, rr: float) -> np.ndarray:
    """mV: QRS (R 1.0 mV, narrow Gaussian) and a T wave, scaled to the beat interval."""
    x = np.zeros_like(t)
    w = 0.012 * rr
    for b in beats:
        sel = (t > b - 0.3 * rr) & (t < b + 0.6 * rr)
        tt = t[sel] - b
        x[sel] += (1.0 * np.exp(-0.5 * (tt / w) ** 2) - 0.15 * np.exp(-0.5 * ((tt + 2.5 * w) / w) ** 2)
                   + 0.25 * np.exp(-0.5 * ((tt - 0.3 * rr) / (0.06 * rr)) ** 2))
    return x


def bp(t: np.ndarray, beats: np.ndarray, sys_: float, dia: float) -> np.ndarray:
    """mmHg: from diastolic at each beat, a fast rise to systolic and an exponential decay back."""
    x = np.full_like(t, dia)
    edges = np.r_[beats, beats[-1] + np.median(np.diff(beats))]
    for b0, b1 in zip(edges[:-1], edges[1:]):
        sel = (t >= b0) & (t < b1)
        ph = (t[sel] - b0) / (b1 - b0)
        up = np.clip(ph / 0.15, 0, 1)
        shape = np.where(ph < 0.15, np.sin(0.5 * np.pi * up), np.exp(-(ph - 0.15) / 0.35))
        shape = (shape - np.exp(-0.85 / 0.35)) / (1 - np.exp(-0.85 / 0.35))
        x[sel] = dia + (sys_ - dia) * np.clip(shape, 0, 1)
    return x


def resp(t: np.ndarray, rate_bpm: float) -> np.ndarray:
    return np.sin(2 * np.pi * rate_bpm / 60.0 * t)


def stim(t: np.ndarray, trains: list[dict], pulse_s: float) -> np.ndarray:
    """A stimulus monitor: amplitude (mA) during each pulse, 0 otherwise."""
    x = np.zeros_like(t)
    for tr in trains:
        for k in range(int(round(tr["duration"] * tr["frequency_hz"]))):
            p0 = tr["onset"] + k / tr["frequency_hz"]
            x[(t >= p0) & (t < p0 + pulse_s)] = tr["amplitude_ma"]
    return x


def recording(fs: float, duration: float, hr: float, sys_: float, dia: float, rr_resp: float, trains: list,
              bpm_per_ma: float, rng) -> dict:
    t = np.arange(int(round(duration * fs))) / fs
    beats = beat_times(duration, hr, trains, bpm_per_ma, rng)
    return {"t": t, "fs": fs, "beats": beats, "ecg": ecg(t, beats, 60.0 / hr), "bp": bp(t, beats, sys_, dia),
            "resp": resp(t, rr_resp), "stim": stim(t, trains, max(1.0 / fs, 0.002)), "trains": trains,
            "truth": {"hr_bpm": hr, "sbp_mmhg": sys_, "dbp_mmhg": dia, "resp_rate_bpm": rr_resp,
                      "r_peaks_s": [round(float(b), 6) for b in beats], "trains": trains, "bpm_per_ma": bpm_per_ma}}


# ------------------------------------------------------------------------------------------- writers
def write_edf(path: Path, signals: list[dict], fs: float, duration_s: int, start: datetime, annotations: list):
    """EDF+C, one data record per second, int16, with an 'EDF Annotations' signal (onset, duration, text)."""
    rec_dur = 1
    n_rec = duration_s // rec_dur
    ann_n = 200  # int16 samples = 400 bytes of TALs per record
    labels = [s["label"] for s in signals] + ["EDF Annotations"]
    ns = len(labels)
    nsamp = [int(fs * rec_dur)] * len(signals) + [ann_n]

    def f(s, n):
        s = str(s)[:n]
        return s + " " * (n - len(s))

    hdr = f("0", 8) + f("X X X X", 80) + f(f"Startdate {start.strftime('%d-%b-%Y').upper()} X X X", 80)
    hdr += start.strftime("%d.%m.%y") + start.strftime("%H.%M.%S") + f(256 * (ns + 1), 8) + f("EDF+C", 44)
    hdr += f(n_rec, 8) + f(rec_dur, 8) + f(ns, 4)
    hdr += "".join(f(x, 16) for x in labels)
    hdr += "".join(f(s.get("transducer", ""), 80) for s in signals) + f("", 80)
    hdr += "".join(f(s["unit"], 8) for s in signals) + f("", 8)
    hdr += "".join(f(f"{s['pmin']:g}", 8) for s in signals) + f(-1, 8)
    hdr += "".join(f(f"{s['pmax']:g}", 8) for s in signals) + f(1, 8)
    hdr += "".join(f(-32768, 8) for _ in signals) + f(-32768, 8)
    hdr += "".join(f(32767, 8) for _ in signals) + f(32767, 8)
    hdr += "".join(f("", 80) for _ in range(ns))
    hdr += "".join(f(n, 8) for n in nsamp)
    hdr += "".join(f("", 32) for _ in range(ns))
    assert len(hdr) == 256 * (ns + 1)
    digs = []
    for s in signals:
        g = 65535.0 / (s["pmax"] - s["pmin"])
        d = np.round((np.clip(s["data"], s["pmin"], s["pmax"]) - s["pmin"]) * g - 32768).astype(np.int64)
        digs.append(np.clip(d, -32768, 32767).astype("<i2"))
    with open(path, "wb") as fh:
        fh.write(hdr.encode("ascii"))
        for r in range(n_rec):
            for d, n in zip(digs, nsamp):
                fh.write(d[r * n:(r + 1) * n].tobytes())
            tal = f"+{r * rec_dur}\x14\x14\x00"
            for on, du, text in annotations:
                if r * rec_dur <= on < (r + 1) * rec_dur:
                    tal += f"+{on:g}" + (f"\x15{du:g}" if du else "") + f"\x14{text}\x14\x00"
            b = tal.encode("latin-1")
            assert len(b) <= 2 * ann_n
            fh.write(b + b"\x00" * (2 * ann_n - len(b)))
    return digs


def write_wfdb(folder: Path, name: str, fs: float, sigs: list[dict], ann_samples: np.ndarray, ann_ext: str = "atr"):
    """Format 16 record (.hea + .dat) and an MIT-format beat annotation file (all 'N')."""
    digs = []
    for s in sigs:
        d = np.clip(np.round(s["data"] * s["gain"] + s.get("baseline", 0)), -32767, 32767).astype("<i2")
        digs.append(d)
    n = digs[0].size
    inter = np.column_stack(digs).astype("<i2")
    (folder / f"{name}.dat").write_bytes(inter.tobytes())
    lines = [f"{name} {len(sigs)} {fs:g} {n}"]
    for s, d in zip(sigs, digs):
        chk = int(d.astype(np.int64).sum()) % 65536
        chk = chk - 65536 if chk >= 32768 else chk
        unit = f"/{s['unit']}" if s.get("unit") else ""
        lines.append(f"{name}.dat 16 {s['gain']:g}({s.get('baseline', 0)}){unit} 16 0 {int(d[0])} {chk} 0 {s['label']}")
    lines.append("# Age: 31 Sex: M")
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")
    out = bytearray()
    prev = 0
    for s in ann_samples:
        diff = int(s) - prev
        prev = int(s)
        if diff > 1023:
            out += struct.pack("<H", (59 << 10))
            out += struct.pack("<H", (diff >> 16) & 0xFFFF) + struct.pack("<H", diff & 0xFFFF)
            diff = 0
        out += struct.pack("<H", (1 << 10) | diff)
    out += b"\x00\x00"
    (folder / f"{name}.{ann_ext}").write_bytes(bytes(out))
    return digs


def write_labchart_mat(path: Path, blocks: list[dict], titles: list[str], units: list[str], comments: list):
    """A LabChart 'Export > MATLAB' file: data, datastart, dataend, samplerate, titles, unittext, unittextmap,
    blocktimes, firstsampleoffset, com, comtext, tickrate."""
    from scipy.io import savemat

    nch, nbl = len(titles), len(blocks)
    data, ds, de, sr = [], np.zeros((nch, nbl)), np.zeros((nch, nbl)), np.zeros((nch, nbl))
    pos = 1
    for b, blk in enumerate(blocks):
        for c in range(nch):
            x = blk["data"][c]
            ds[c, b], de[c, b], sr[c, b] = pos, pos + x.size - 1, blk["fs"]
            data.append(x)
            pos += x.size
    utext = sorted(set(units))
    umap = np.array([[utext.index(units[c]) + 1 for _ in range(nbl)] for c in range(nch)], float)
    width = max(len(s) for s in titles + utext + [c[3] for c in comments] + ["x"])
    pad = lambda xs: np.array([s.ljust(width) for s in xs])  # noqa: E731
    com = np.array([[c[0], c[1] + 1, c[2], 1, i + 1] for i, c in enumerate(comments)], float)
    savemat(path, {"data": np.concatenate(data)[None, :], "datastart": ds, "dataend": de, "samplerate": sr,
                   "titles": pad(titles), "unittext": pad(utext), "unittextmap": umap,
                   "blocktimes": np.array([[b["datenum"] for b in blocks]]), "firstsampleoffset": np.zeros((nch, nbl)),
                   "com": com, "comtext": pad([c[3] for c in comments]),
                   "tickrate": np.array([[b["fs"] for b in blocks]])})


def datenum(d: datetime) -> float:
    from datetime import timedelta
    frac = (d - datetime(d.year, d.month, d.day)) / timedelta(days=1)
    return d.toordinal() + 366 + frac


# ------------------------------------------------------------------------------------------- fixture
def main(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    rec_dir = out / "recordings"
    if rec_dir.exists():
        shutil.rmtree(rec_dir)
    rec_dir.mkdir(parents=True)
    truth: dict = {"sources": {}}
    start = datetime(2024, 3, 5, 9, 30, 0)

    # 1. human, EDF+: ECG (mV), BP (mmHg), RESP, a stimulus monitor (mA), and annotations
    h_trains = [{"onset": 30.0, "duration": 10.0, "amplitude_ma": a, "frequency_hz": 20.0, "pulse_width_us": 250}
                for a in (0.5,)] + [
               {"onset": 60.0, "duration": 10.0, "amplitude_ma": 1.0, "frequency_hz": 20.0, "pulse_width_us": 250},
               {"onset": 90.0, "duration": 10.0, "amplitude_ma": 2.0, "frequency_hz": 20.0, "pulse_width_us": 250}]
    H = recording(250.0, 120.0, 60.0, 120.0, 80.0, 15.0, h_trains, 5.0, rng)
    d = rec_dir / "sub-h01" / "ses-1"
    d.mkdir(parents=True)
    ann = [(30.0, 10.0, "stim 0.5 mA"), (60.0, 10.0, "stim 1.0 mA"), (90.0, 10.0, "stim 2.0 mA"), (5.5, 0, "eyes closed")]
    sigs = [{"label": "ECG II", "unit": "mV", "pmin": -5, "pmax": 5, "data": H["ecg"]},
            {"label": "ABP", "unit": "mmHg", "pmin": 0, "pmax": 300, "data": H["bp"]},
            {"label": "Resp belt", "unit": "", "pmin": -2, "pmax": 2, "data": H["resp"]},
            {"label": "Stim monitor", "unit": "mA", "pmin": -5, "pmax": 5, "data": H["stim"]}]
    write_edf(d / "sub-h01_ses-1_physio.edf", sigs, 250.0, 120, start, ann)
    truth["sources"]["sub-h01/ses-1/sub-h01_ses-1_physio.edf"] = {
        "recording_id": "sub-h01_ses-1_rec-physio", "species": "human", **H["truth"], "fs": 250.0, "n": H["t"].size,
        "channels": {"ecg": ["ecg", "V", 1e-3, True], "bp": ["bp", "mmHg", 1.0, True], "resp": ["resp", "a.u.", 1.0, False],
                     "stim": ["stim", "mA", 1.0, True]},
        "annotations": [a[2] for a in sorted(ann)], "start_time": start.isoformat()}

    # 2. rat, WFDB: ECG (mV), BP (mmHg), RESP (no unit) + beat annotations
    r_trains = [{"onset": 20.0, "duration": 5.0, "amplitude_ma": a, "frequency_hz": 10.0, "pulse_width_us": 200}
                for a in (0.5,)] + [{"onset": 40.0, "duration": 5.0, "amplitude_ma": 1.0, "frequency_hz": 10.0,
                                     "pulse_width_us": 200}]
    Rr = recording(1000.0, 60.0, 350.0, 110.0, 75.0, 80.0, r_trains, 20.0, rng)
    d = rec_dir / "rat-01"
    d.mkdir(parents=True)
    write_wfdb(d, "rat01", 1000.0, [{"label": "ECG", "unit": "mV", "gain": 1000.0, "data": Rr["ecg"]},
                                    {"label": "BP", "unit": "mmHg", "gain": 100.0, "baseline": -9000, "data": Rr["bp"]},
                                    {"label": "RESP", "unit": "", "gain": 1000.0, "data": Rr["resp"]}],
               np.round(Rr["beats"] * 1000.0).astype(int))
    truth["sources"]["rat-01/rat01.hea"] = {
        "recording_id": "sub-rat01_rec-wfdb", "species": "rat", **Rr["truth"], "fs": 1000.0, "n": Rr["t"].size,
        "channels": {"ecg": ["ecg", "V", 1e-3, True], "bp": ["bp", "mmHg", 1.0, True], "resp": ["resp", "a.u.", 1.0, False]},
        "n_annotations": int(Rr["beats"].size), "age": "P31Y", "sex": "M"}

    # 3. human without BP, CSV: Time, ECG (mV), Resp, Stim (empty before the stimulator is on), a text marker
    #    column, a flat channel and a clipped channel
    N = recording(250.0, 90.0, 72.0, 120.0, 80.0, 12.0,
                  [{"onset": 40.0, "duration": 10.0, "amplitude_ma": 1.5, "frequency_hz": 30.0, "pulse_width_us": 100}],
                  5.0, rng)
    import pandas as pd
    st = N["stim"].copy()
    st[N["t"] < 20.0] = np.nan
    clipped = np.clip(2.0 * N["resp"], -1.0, 1.0)
    marks = np.full(N["t"].size, "", dtype=object)
    marks[np.searchsorted(N["t"], 40.0)] = "stim on"
    marks[np.searchsorted(N["t"], 50.0)] = "stim off"
    df = pd.DataFrame({"Time (s)": np.round(N["t"], 4), "ECG (mV)": np.round(N["ecg"], 5), "Resp": np.round(N["resp"], 4),
                       "StimAmp": st, "Temp (degC)": np.full(N["t"].size, 36.5), "Thorax belt": clipped,
                       "Comment": marks})
    d = rec_dir / "sub-h02"
    d.mkdir(parents=True)
    df.to_csv(d / "session-rest.csv", index=False)
    truth["sources"]["sub-h02/session-rest.csv"] = {
        "recording_id": "sub-h02_rec-session-rest", **N["truth"], "fs": 250.0, "n": N["t"].size,
        "channels": {"ecg": ["ecg", "V", 1e-3, True], "resp": ["resp", "a.u.", 1.0, False], "stim": ["stim", "a.u.", 1.0, False],
                     "temp": ["temp", "degC", 1.0, True], "resp-2": ["resp", "a.u.", 1.0, False]},
        "status": {"temp": "flat", "resp-2": "saturated"}, "stim_nan_fraction": float(np.mean(N["t"] < 20.0)),
        "markers": ["stim on", "stim off"]}
    # an identical copy: read once
    shutil.copy(d / "session-rest.csv", d / "session-rest - Copy.csv")

    # 4. pig-style trial-by-trial HDF5: one group per stimulation, column datasets, overlapping windows
    import h5py
    P = recording(1000.0, 140.0, 90.0, 100.0, 60.0, 14.0, [], 0.0, rng)
    fs_n = 5000.0
    tn = np.arange(int(140.0 * fs_n)) / fs_n
    neural = np.round(rng.standard_normal(tn.size) * 10 / 0.195) * 0.195   # an ADC with 0.195 steps
    stim_line = np.full(tn.size, -100.0)
    hr_dev = 90.0 + 2.0 * np.sin(2 * np.pi * 0.1 * P["t"][::2])
    epoch0 = 1_700_000_000.0
    onsets = [10.0, 27.0, 44.0, 100.0]   # windows of 25 s from onset-10: the first three overlap, the last is apart
    params = [(100, 2.0), (500, 2.0), (1000, 10.0), (2000, 10.0)]
    for on, (cur, fr) in zip(onsets, params):
        for k in range(int(5 * fr)):
            p0 = on + k / fr
            stim_line[(tn >= p0) & (tn < p0 + 0.0005)] = 100.0
    d = rec_dir / "sub-p1"
    d.mkdir(parents=True)
    with h5py.File(d / "subject-p1.hdf5", "w") as f:
        g0 = f.create_group("subject-p1")
        g0.attrs["date"] = "2023-11-14"
        for on, (cur, fr) in zip(onsets, params):
            w0, w1 = on - 10.0, on + 15.0
            name = datetime.fromtimestamp(epoch0 + on, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
            g = g0.create_group(name)
            g.attrs.update({"current_uA": cur, "frequency_Hz": fr, "pulse_width_us": 500, "duration_s": 5.0,
                            "polarity": "cathodic", "stim_shape": "monophasic"})
            i0, i1 = int(round(w0 * 1000)), int(round(w1 * 1000))
            e = np.column_stack([epoch0 + P["t"][i0:i1], P["ecg"][i0:i1] * 1e-3, P["ecg"][i0:i1] * 2e-3])
            ds = g.create_dataset("ecg", data=e)
            ds.attrs["_column_names"] = np.array(["timestamps", "raw_ecg", "processed_ecg"], dtype=object)
            ds.attrs["frequency_hz"] = 1000.0
            j0, j1 = i0 // 2, i1 // 2
            ds = g.create_dataset("hr", data=np.column_stack([epoch0 + P["t"][::2][j0:j1], hr_dev[j0:j1]]))
            ds.attrs["_column_names"] = np.array(["timestamps", "hr"], dtype=object)
            ds.attrs["frequency_hz"] = 500.0
            k0, k1 = int(round(w0 * fs_n)), int(round(w1 * fs_n))
            ds = g.create_dataset("neural", data=np.column_stack([epoch0 + tn[k0:k1], neural[k0:k1], stim_line[k0:k1]]))
            ds.attrs["_column_names"] = np.array(["timestamps", "record1", "stim_timing"], dtype=object)
            ds.attrs["frequency_hz"] = fs_n
    truth["sources"]["sub-p1/subject-p1.hdf5"] = {
        "recording_id": "sub-p1_rec-hdf5", "n_trials": len(onsets), "n_segments": 2,
        "trial_params": [{"current_uA": c, "frequency_Hz": fr} for c, fr in params],
        "channels": {"ecg": ["ecg", "a.u.", 1.0, False], "ecg-2": ["ecg", "a.u.", 1.0, False], "hr": ["hr", "bpm", 1.0, True],
                     "nerve": ["nerve", "a.u.", 1.0, False], "stim": ["stim", "a.u.", 1.0, False]},
        "start_epoch": epoch0, "nerve_step": 0.195, "ecg_samples": int(round((59.0 + 25.0) * 1000))}

    # 5. LabChart MATLAB export: two blocks, ECG (mV) and BP (mmHg), comments
    L1 = recording(400.0, 30.0, 65.0, 125.0, 85.0, 15.0, [], 0.0, rng)
    L2 = recording(400.0, 20.0, 65.0, 125.0, 85.0, 15.0, [], 0.0, rng)
    t_l = datetime(2024, 5, 1, 14, 0, 0)
    from datetime import timedelta
    blocks = [{"fs": 400.0, "datenum": datenum(t_l), "data": [L1["ecg"], L1["bp"]]},
              {"fs": 400.0, "datenum": datenum(t_l + timedelta(seconds=60)), "data": [L2["ecg"], L2["bp"]]}]
    d = rec_dir / "labchart"
    d.mkdir(parents=True)
    write_labchart_mat(d / "dog7_baseline.mat", blocks, ["ECG", "Arterial pressure"], ["mV", "mmHg"],
                       [(-1, 0, 4000, "drug given"), (2, 1, 800, "flush")])
    truth["sources"]["labchart/dog7_baseline.mat"] = {
        "recording_id": "sub-dog7-baseline_rec-mat", "n_segments": 2, "fs": 400.0, "markers": {"drug given": 10.0, "flush": 62.0},
        "start_time": t_l.isoformat(), "sbp_mmhg": 125.0, "dbp_mmhg": 85.0,
        "channels": {"ecg": ["ecg", "V", 1e-3, True], "bp": ["bp", "mmHg", 1.0, True]}}

    # 6. WAV: one uncalibrated ECG channel (needs channels-map.csv to be typed), 500 Hz int16
    from scipy.io import wavfile
    W = recording(500.0, 40.0, 70.0, 120.0, 80.0, 15.0, [], 0.0, rng)
    d = rec_dir / "sub-h03" / "ses-B"
    d.mkdir(parents=True)
    wavfile.write(d / "ecg-1.wav", 500, np.round(W["ecg"] * 8000).astype(np.int16))
    truth["sources"]["sub-h03/ses-B/ecg-1.wav"] = {"recording_id": "sub-h03_ses-B_rec-ecg-1", "fs": 500.0, **W["truth"],
                                                   "channels": {"ecg": ["ecg", "a.u.", 1.0, False]}}

    # 7. XLSX: slow respiration and PPG at 20 Hz
    X = recording(20.0, 30.0, 70.0, 120.0, 80.0, 18.0, [], 0.0, rng)
    pd.DataFrame({"time": X["t"], "Airflow": X["resp"], "Pleth": X["bp"] / 100}).to_excel(
        rec_dir / "sub-h04_breathing.xlsx", index=False)
    truth["sources"]["sub-h04_breathing.xlsx"] = {"recording_id": "sub-h04_rec-breathing", "fs": 20.0,
                                                 "channels": {"resp": ["resp", "a.u.", 1.0, False],
                                                              "ppg": ["ppg", "a.u.", 1.0, False]}}

    # 8. a .mat without a recognisable layout, described by layout.yaml
    from scipy.io import savemat
    M = recording(200.0, 20.0, 80.0, 120.0, 80.0, 15.0, [], 0.0, rng)
    d = rec_dir / "custom"
    d.mkdir()
    savemat(d / "session_custom.mat", {"sig": np.column_stack([M["ecg"], M["bp"]]), "srate_hz": 200.0})
    (d / "layout.yaml").write_text(
        'files: "session_custom.mat"\nrate: srate_hz\nsignals:\n  - path: sig\n    names: [ECG, ABP]\n'
        "    units: [mV, mmHg]\n")
    truth["sources"]["custom/session_custom.mat"] = {"recording_id": "sub-session-custom_rec-mat", "fs": 200.0,
                                                    "channels": {"ecg": ["ecg", "V", 1e-3, True], "bp": ["bp", "mmHg", 1.0, True]}}

    # 9. deferred formats and the mapping files
    (rec_dir / "deferred").mkdir()
    (rec_dir / "deferred" / "trial.smrx").write_bytes(b"S2 64-bit placeholder")
    (rec_dir / "deferred" / "bladder.adicht").write_bytes(b"LabChart placeholder")
    truth["deferred"] = ["deferred/trial.smrx", "deferred/bladder.adicht"]
    (rec_dir / "channels-map.csv").write_text("pattern,type,unit,scale,site,name,path_glob\n"
                                              "^channel 1$,ecg,,,lead II,,*.wav\n")
    (rec_dir / "recordings-map.csv").write_text("path_glob,subject,session,label,species\n"
                                                "rat-01/*,rat01,,,rat\n"
                                                "labchart/*,dog7,,baseline,dog\n")
    truth["species"] = {"sub-rat01_rec-wfdb": "rat", "sub-dog7_rec-baseline": "dog"}
    truth["sources"]["labchart/dog7_baseline.mat"]["recording_id"] = "sub-dog7_rec-baseline"

    # 10. a BIDS dataset with physiological recordings only (no imaging): sidecar inheritance, StartTime, several
    #     recording-<label> files and a _stim file in one run, task events, participants.tsv
    import gzip
    B = rec_dir / "bids"
    (B / "sub-b01" / "func").mkdir(parents=True)
    (B / "dataset_description.json").write_text(json.dumps({"Name": "synthetic physio", "BIDSVersion": "1.10.0"}))
    (B / "participants.tsv").write_text("participant_id\tspecies\tage\tsex\nsub-b01\thomo sapiens\t30\tF\n")
    (B / "task-rest_physio.json").write_text(json.dumps({   # inherited by run 1 (no sidecar next to it)
        "SamplingFrequency": 100, "StartTime": -2.0, "Columns": ["cardiac", "respiratory", "trigger"],
        "cardiac": {"Description": "finger pulse oximeter"}, "respiratory": {"Description": "belt"}}))

    def tsvgz(path, cols):
        with gzip.open(path, "wt") as fh:
            for row in np.column_stack(cols):
                fh.write("\t".join(f"{v:.5g}" for v in row) + "\n")

    b1 = recording(100.0, 60.0, 64.0, 120.0, 80.0, 14.0, [], 0.0, rng)
    trig = (np.arange(b1["t"].size) % 200 < 5).astype(float)
    tsvgz(B / "sub-b01" / "func" / "sub-b01_task-rest_run-1_physio.tsv.gz", [b1["bp"] / 100, b1["resp"], trig])
    b2 = recording(500.0, 60.0, 64.0, 120.0, 80.0, 14.0, [], 0.0, rng)
    f2 = B / "sub-b01" / "func"
    tsvgz(f2 / "sub-b01_task-rest_run-2_recording-ecg_physio.tsv.gz", [b2["ecg"]])
    (f2 / "sub-b01_task-rest_run-2_recording-ecg_physio.json").write_text(json.dumps({
        "SamplingFrequency": 500, "StartTime": -1.0, "Columns": ["cardiac"],
        "cardiac": {"Description": "ECG lead II", "Units": "mV"}}))
    tsvgz(f2 / "sub-b01_task-rest_run-2_recording-resp_physio.tsv.gz", [b2["resp"][::20]])
    (f2 / "sub-b01_task-rest_run-2_recording-resp_physio.json").write_text(json.dumps({
        "SamplingFrequency": 25, "StartTime": 0.0, "Columns": ["respiratory"]}))
    on = np.where((b2["t"] > 20) & (b2["t"] < 25), 1.0, 0.0)[::5]
    tsvgz(f2 / "sub-b01_task-rest_run-2_stim.tsv.gz", [on * 5.0, 10 + 90 * on])
    (f2 / "sub-b01_task-rest_run-2_stim.json").write_text(json.dumps({
        "SamplingFrequency": 100, "StartTime": -1.0, "Columns": ["trigger", "luminance"], "trigger": {"Units": "V"},
        "luminance": {"Units": "cd/m2", "Description": "screen luminance (the presented stimulus)"}}))
    (f2 / "sub-b01_task-rest_run-2_events.tsv").write_text("onset\tduration\ttrial_type\n10.0\t5.0\tbreath-hold\n")
    truth["sources"]["bids/sub-b01/func/sub-b01_task-rest_run-1_physio.tsv.gz"] = {
        "recording_id": "sub-b01_rec-rest-run1", "species": "human", "age": "P30Y", "sex": "F", "task": "rest", "run": "1",
        "channels": {"ppg": ["ppg", "a.u.", 1.0, False], "resp": ["resp", "a.u.", 1.0, False],
                     "other": ["other", "a.u.", 1.0, False]}}
    truth["sources"]["bids/sub-b01/func/sub-b01_task-rest_run-2_recording-ecg_physio.tsv.gz"] = {
        "recording_id": "sub-b01_rec-rest-run2", "task": "rest", "run": "2",
        "channels": {"ecg": ["ecg", "V", 1e-3, True], "resp": ["resp", "a.u.", 1.0, False], "stim": ["stim", "V", 1.0, True],
                     "other": ["other", "cd/m2", 1.0, True]},
        "starts": {"ecg": 0.0, "resp": 1.0, "stim": 0.0}, "rates": {"ecg": 500.0, "resp": 25.0, "stim": 100.0},
        "events": {"breath-hold": 11.0}}

    # 11. optional: ABF (pyabf) and NWB (pynwb) inputs
    try:
        import pyabf.abfWriter as aw
        A = recording(2000.0, 10.0, 300.0, 100.0, 70.0, 60.0, [], 0.0, rng)
        sweeps = np.stack([A["ecg"][i * 5000:(i + 1) * 5000] for i in range(4)])
        (rec_dir / "abf").mkdir()
        aw.writeABF1(sweeps, str(rec_dir / "abf" / "mouse_ecg.abf"), sampleRateHz=2000.0, units="mV")
        truth["sources"]["abf/mouse_ecg.abf"] = {"n_segments": 4, "fs": 2000.0, "data": "ecg sweeps"}
    except ImportError:
        pass
    try:
        _write_nwb(rec_dir / "nwb", rng)
        truth["sources"]["nwb/sub-m1_ses-1.nwb"] = {"recording_id": "sub-m1_ses-1_rec-nwb", "species": "mouse",
                                                   "channels": {"ecg": ["ecg", "V", 1.0, True], "bp": ["bp", "mmHg", 1.0, True]}}
    except ImportError:
        pass
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"fixture written to {rec_dir} ({sum(1 for _ in rec_dir.rglob('*') if _.is_file())} files)")


def _write_nwb(d: Path, rng) -> None:
    from datetime import timezone

    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.file import Subject

    d.mkdir()
    R = recording(1000.0, 20.0, 600.0, 105.0, 75.0, 150.0, [], 0.0, rng)
    nwb = NWBFile(session_description="synthetic mouse", identifier="m1", session_start_time=datetime(2024, 1, 2, 3, 4, 5,
                  tzinfo=timezone.utc), subject=Subject(subject_id="m1", species="Mus musculus", sex="F", age="P90D"))
    nwb.add_acquisition(TimeSeries(name="ECG", data=R["ecg"] * 1e-3, unit="V", rate=1000.0))
    nwb.add_acquisition(TimeSeries(name="ArterialPressure", data=R["bp"], unit="mmHg", rate=1000.0))
    nwb.add_trial_column("amplitude_ma", "stimulus amplitude")
    nwb.add_trial(start_time=5.0, stop_time=8.0, amplitude_ma=0.4)
    with NWBHDF5IO(str(d / "sub-m1_ses-1.nwb"), "w") as io:
        io.write(nwb)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixture"))
