"""The synthetic fixture (tests/make_fixture.py) through the whole app, without Docker."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from processor import channels as C
from processor import metadata as M

REPO = Path(__file__).resolve().parent.parent


def run_app(input_dir: Path, output_dir: Path, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "INPUT_DIR": str(input_dir), "OUTPUT_DIR": str(output_dir), "TMPDIR": str(output_dir.parent / "tmp"),
         "THREADS": "1", **env}
    return subprocess.run([sys.executable, "-m", "processor.main"], cwd=REPO, env=e, capture_output=True, text=True,
                          timeout=600)


@pytest.fixture(scope="session")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("fx")
    subprocess.run([sys.executable, "-m", "tests.make_fixture", str(base / "fixture")], cwd=REPO, check=True)
    out = base / "out"
    p = run_app(base / "fixture" / "recordings", out)
    assert p.returncode == 0, p.stdout + p.stderr
    truth = json.loads((base / "fixture" / "truth.json").read_text())
    physio = out / "physio"
    recs = {r["recording_id"]: r for r in csv.DictReader(open(physio / "recordings.csv"))}
    chans = {}
    for r in csv.DictReader(open(physio / "channels.csv")):
        chans.setdefault(r["recording_id"], {})[r["channel"]] = r
    events = list(csv.DictReader(open(physio / "events.tsv"), delimiter="\t"))
    prov = json.loads((physio / "provenance.json").read_text())
    return {"out": out, "physio": physio, "truth": truth, "recs": recs, "chans": chans, "events": events,
            "prov": prov, "log": p.stdout}


def nwb_series(run, rid: str, name: str):
    from pynwb import NWBHDF5IO

    io = NWBHDF5IO(str(run["physio"] / run["recs"][rid]["path"]), "r")
    nwb = io.read()
    ts = nwb.acquisition[name]
    data = ts.data[:] * ts.conversion + ts.offset
    times = ts.timestamps[:] if ts.timestamps is not None else ts.starting_time + np.arange(data.size) / ts.rate
    return nwb, ts, data, times, io


def test_every_readable_source_is_a_recording(run):
    expected = {v["recording_id"] for v in run["truth"]["sources"].values() if "recording_id" in v}
    assert expected <= set(run["recs"])
    for rid, sp in run["truth"]["species"].items():
        assert run["recs"][rid]["species"] == sp
    assert run["recs"]["sub-m1_ses-1_rec-nwb"]["species"] == "mouse"
    assert run["recs"]["sub-rat01_rec-wfdb"]["age"] == "P31Y" and run["recs"]["sub-rat01_rec-wfdb"]["sex"] == "M"
    assert run["recs"]["sub-h01_ses-1_rec-physio"]["start_time"] == "2024-03-05T09:30:00"
    for r in run["recs"].values():
        assert (run["physio"] / r["path"]).is_file()


def test_channel_types_units_and_calibration(run):
    for src, t in run["truth"]["sources"].items():
        rid = t.get("recording_id")
        if not rid or "channels" not in t:
            continue
        got = run["chans"][rid]
        for name, (typ, unit, scale, cal) in t["channels"].items():
            assert name in got, (rid, name, list(got))
            r = got[name]
            assert (r["type"], r["unit"], r["calibrated"]) == (typ, unit, "true" if cal else "false"), (rid, name, r)
            assert float(r["scale"]) == pytest.approx(scale), (rid, name)
    assert run["chans"]["sub-h03_ses-B_rec-ecg-1"]["ecg"]["typing"] == "mapping"
    assert run["chans"]["sub-h03_ses-B_rec-ecg-1"]["ecg"]["site"] == "lead II"


def test_checks_flag_flat_and_clipped_channels(run):
    got = run["chans"]["sub-h02_rec-session-rest"]
    assert got["temp"]["status"] == "flat"
    assert got["resp-2"]["status"] == "saturated"
    assert got["ecg"]["status"] == "good" and got["resp"]["status"] == "good"
    # a sine quantised at its peaks is not clipping
    assert run["chans"]["sub-rat01_rec-wfdb"]["resp"]["status"] == "good"


def test_edf_values_and_r_peaks(run):
    t = run["truth"]["sources"]["sub-h01/ses-1/sub-h01_ses-1_physio.edf"]
    nwb, ts, ecg, times, io = nwb_series(run, "sub-h01_ses-1_rec-physio", "ecg")
    assert ts.rate == 250 and ts.data.dtype == np.int16
    peaks = np.round(np.asarray(t["r_peaks_s"]) * 250).astype(int)
    peaks = peaks[(peaks > 2) & (peaks < ecg.size - 3)]
    # every R peak is the local maximum, about 1 mV
    for p in peaks[:20]:
        assert np.argmax(ecg[p - 3:p + 4]) in (2, 3, 4)
    assert np.median(ecg[peaks]) == pytest.approx(1e-3, rel=0.1)
    _, _, bp, _, io2 = nwb_series(run, "sub-h01_ses-1_rec-physio", "bp")
    assert bp.max() == pytest.approx(t["sbp_mmhg"], abs=1.0)
    assert np.percentile(bp, 1) == pytest.approx(t["dbp_mmhg"], abs=2.0)
    io.close()
    io2.close()


def test_wfdb_digital_values_and_annotations(run):
    t = run["truth"]["sources"]["rat-01/rat01.hea"]
    _, ts, bp, _, io = nwb_series(run, "sub-rat01_rec-wfdb", "bp")
    assert ts.data.dtype == np.int16
    assert bp.max() == pytest.approx(t["sbp_mmhg"], abs=0.05)
    assert bp.min() == pytest.approx(t["dbp_mmhg"], abs=0.05)
    io.close()
    ann = [e for e in run["events"] if e["recording_id"] == "sub-rat01_rec-wfdb"]
    assert len(ann) == t["n_annotations"]
    on = np.array([float(e["onset"]) for e in ann])
    assert np.max(np.abs(on - np.asarray(t["r_peaks_s"]))) < 0.0011
    assert {e["trial_type"] for e in ann} == {"annotation"} and ann[0]["label"] == "N"


def test_hdf5_trials_overlaps_parameters(run):
    t = run["truth"]["sources"]["sub-p1/subject-p1.hdf5"]
    rid = t["recording_id"]
    assert run["recs"][rid]["n_segments"] == str(t["n_segments"])
    assert run["recs"][rid]["start_time"] == "2023-11-14T22:13:20"
    nwb, ts, ecg, times, io = nwb_series(run, rid, "ecg")
    assert ecg.size == t["ecg_samples"]          # overlapping windows written once
    assert np.all(np.diff(times) > 0)
    nerve = nwb.acquisition["nerve"]
    assert nerve.data.dtype == np.int16 and nerve.conversion == pytest.approx(t["nerve_step"])
    st = nwb.intervals["stimulation_source"].to_dataframe()
    assert len(st) == t["n_trials"]
    assert list(st["current_uA"]) == [p["current_uA"] for p in t["trial_params"]]
    assert set(st["polarity"]) == {"cathodic"}
    # the channels with the same time base share one timestamps dataset
    assert nwb.acquisition["stim"].timestamps is nerve.timestamps or \
        nwb.acquisition["stim"].timestamp_link
    io.close()


def test_csv_gaps_markers_and_duplicates(run):
    t = run["truth"]["sources"]["sub-h02/session-rest.csv"]
    _, ts, stim, _, io = nwb_series(run, "sub-h02_rec-session-rest", "stim")
    assert np.isnan(stim).mean() == pytest.approx(t["stim_nan_fraction"], abs=1e-3)
    io.close()
    mk = [e for e in run["events"] if e["recording_id"] == "sub-h02_rec-session-rest" and e["trial_type"] == "marker"]
    assert [e["label"] for e in mk] == t["markers"]
    art = [e for e in run["events"] if e["recording_id"] == "sub-h02_rec-session-rest" and e["trial_type"] == "artifact"]
    assert art and all(e["source"] == "qc" and e["label"].startswith("resp-2 clipped") for e in art)
    assert float(mk[0]["onset"]) == pytest.approx(40.0, abs=0.005)
    dup = run["prov"]["recordings"]["sub-h02/session-rest - Copy.csv"]
    assert dup["status"] == "skipped" and "identical" in dup["reason"]


def test_labchart_blocks_and_comments(run):
    t = run["truth"]["sources"]["labchart/dog7_baseline.mat"]
    rid = t["recording_id"]
    assert run["recs"][rid]["n_segments"] == "2"
    assert run["recs"][rid]["start_time"].startswith("2024-05-01T14:00:00")
    mk = {e["label"]: float(e["onset"]) for e in run["events"] if e["recording_id"] == rid}
    for k, v in t["markers"].items():
        assert mk[k] == pytest.approx(v, abs=0.01)


def test_deferred_formats_say_how_to_export(run):
    p = run["prov"]["recordings"]
    assert p["deferred/trial.smrx"]["status"] == "skipped" and "sonpy" in p["deferred/trial.smrx"]["reason"]
    assert ".smr" in p["deferred/trial.smrx"]["reason"]
    assert p["deferred/bladder.adicht"]["status"] == "skipped" and "GPL" in p["deferred/bladder.adicht"]["reason"]


def test_nwb_input_and_trials(run):
    nwb, ts, ecg, _, io = nwb_series(run, "sub-m1_ses-1_rec-nwb", "ecg")
    assert nwb.subject.species == "Mus musculus" and nwb.subject.age == "P90D"
    assert len(nwb.intervals["stimulation_source"].to_dataframe()) == 1
    io.close()


def test_tables_provenance_summary(run):
    head = open(run["physio"] / "events.tsv").readline().rstrip("\n").split("\t")
    assert head[:20] == ["recording_id", "onset", "duration", "trial_type", "event_id", "amplitude", "amplitude_unit",
                         "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s", "n_pulses",
                         "polarity", "waveform", "site", "contact", "condition", "label", "source"]
    head = open(run["physio"] / "recordings.csv").readline().strip().split(",")
    assert head == ["recording_id", "subject", "session", "label", "path", "species", "start_time", "duration_s",
                    "recorded_s", "n_segments", "channel_types", "source_format", "source_path", "metadata_source",
                    "age", "sex", "task", "run"]
    prov = run["prov"]
    for k in ("app", "version", "variant", "tools", "parameters", "inputs", "started", "finished", "wall_seconds",
              "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"):
        assert k in prov
    qc = run["out"] / "qc" / "physio-import"
    summ = json.loads((qc / "summary.json").read_text())
    assert summ["app"] == "physio-import" and len(summ["figures"]) == len(run["recs"])
    for f in summ["figures"]:
        assert (qc / f["path"]).is_file() and f["caption"]
    insp = list(csv.DictReader(open(qc / "nwbinspector.csv")))
    assert not [i for i in insp if i["importance"] == "CRITICAL" and not i["check"].startswith("check_subject")]


def test_bids_physio(run):
    t1 = run["truth"]["sources"]["bids/sub-b01/func/sub-b01_task-rest_run-1_physio.tsv.gz"]
    r1 = run["recs"][t1["recording_id"]]
    assert (r1["species"], r1["age"], r1["sex"], r1["task"], r1["run"]) == ("human", "P30Y", "F", "rest", "1")
    assert r1["source_format"] == "bids-physio"
    t2 = run["truth"]["sources"]["bids/sub-b01/func/sub-b01_task-rest_run-2_recording-ecg_physio.tsv.gz"]
    rid = t2["recording_id"]
    assert set(run["chans"][rid]) == set(t2["channels"])     # three files of one run, one recording
    for name, st in t2["starts"].items():
        nwb, ts, x, times, io = nwb_series(run, rid, name)
        assert times[0] == pytest.approx(st) and ts.rate == t2["rates"][name]
        io.close()
    ev = {e["label"]: float(e["onset"]) for e in run["events"] if e["recording_id"] == rid}
    assert ev["breath-hold"] == pytest.approx(t2["events"]["breath-hold"])
    # BIDS sidecars and events tables are not recordings
    assert not [k for k in run["prov"]["recordings"] if k.endswith(("_events.tsv", ".json", "participants.tsv"))]


def test_exit_codes(tmp_path):
    (tmp_path / "empty").mkdir()
    assert run_app(tmp_path / "empty", tmp_path / "o1").returncode == 1
    assert run_app(tmp_path / "empty", tmp_path / "o2", SPECIES="cat").returncode == 2
    (tmp_path / "only").mkdir()
    (tmp_path / "only" / "x.smrx").write_bytes(b"x")
    assert run_app(tmp_path / "only", tmp_path / "o3").returncode == 1


@pytest.mark.parametrize("name,leaf,typ", [
    ("ECG1", "", "ecg"), ("Lead II", "", "ecg"), ("3 ECG", "", "ecg"), ("ecg/raw_ecg", "raw_ecg", "ecg"),
    ("NIBP", "", "bp"), ("Finapres", "", "bp"), ("Heart rate", "", "hr"), ("HR (ECG)", "", "hr"), ("hr/hr", "hr", "hr"),
    ("4 StimAmp", "", "stim"), ("5 CtrlAmp", "", "stim"), ("neural/stim_timing", "stim_timing", "stim"),
    ("neural/record1", "record1", "nerve"), ("Vagus cuff", "", "nerve"), ("RESP", "", "resp"), ("Airflow", "", "resp"),
    ("Pleth", "", "ppg"), ("Bladder pressure", "", "pressure"), ("EMG gastric", "", "emg"), ("Temp", "", "temp"),
    ("heart sound", "", None), ("capacity", "", None),
])
def test_name_heuristics(name, leaf, typ):
    assert C.type_by_name(name, leaf) == typ


def test_calibration():
    assert C.calibrate("ecg", "uV").scale == pytest.approx(1e-6)
    assert C.calibrate("bp", "kPa").scale == pytest.approx(7.500617)
    c = C.calibrate("bp", "V")
    assert not c.calibrated and c.unit == "V"
    assert not C.calibrate("ecg", "").calibrated
    assert C.calibrate("ecg", "", "mV", 0.5).scale == pytest.approx(5e-4)


def test_metadata_helpers():
    assert M.age_iso("4 months") == "P4M" and M.age_iso("20 years") == "P20Y" and M.age_iso("23") == "P23Y"
    assert M.species_code("Sus scrofa") == "pig" and M.species_code("Rattus norvegicus") == "rat"
    assert M.label_from_stem("Perf-Bimodal01-set01-trial01-gain02", "Bimodal01", "set01-trial01-gain02",
                             "Bimodal01-set01-trial01-gain02", "csv") == "csv"
    assert M.label_from_stem("01-A-1", "1001", "A", "1001-A", "wav") == "01-A-1"
    assert M.session_from_perf("1001-A", "1001") == "A"
