# physio-import

Reads cardiovascular and autonomic physiology recordings, as common acquisition systems export them,
into one NWB file per recording plus plain index tables. It is built on other people's readers:
**Neo**, from the NeuralEnsemble community (Samuel Garcia, Andrew Davison and colleagues;
https://github.com/NeuralEnsemble/python-neo; Garcia et al. 2014) for Axon ABF and Spike2 `.smr`;
**WFDB for Python**, from the MIT Laboratory for Computational Physiology (Chen Xie and colleagues;
https://github.com/MIT-LCP/wfdb-python; PhysioNet doi:10.13026/9njx-6322, with Goldberger et al. 2000)
for PhysioNet records; **pyEDFlib**, by Holger Nahrstaedt, Chris Lee-Messer and Teunis van Beelen
(https://github.com/holgern/pyedflib) for EDF/BDF; **bioread**, from the University of
Wisconsin-Madison Center for Healthy Minds (Nate Vack, John Ollinger, Dan Fitch and Mike Davison;
https://github.com/uwmadison-chm/bioread) for BIOPAC `.acq`; **pymatreader** by Dirk Gütlin and
Thomas Hartmann for MATLAB files; and **PyNWB** and **NWB Inspector**, from the Neurodata Without
Borders project (https://github.com/NeurodataWithoutBorders; Rübel et al. 2022) to write and check
NWB. Pennsieve packages these tools; it did not develop them. BIDS physiological recordings are read
as the BIDS specification (BIDS contributors; Gorgolewski et al. 2016) defines them. The demo data
are by Oliver Armitage and colleagues (Pennsieve Discover 349), Shane A. Bender and colleagues
(Discover 655), M. Khawar Ali and colleagues (Discover 406), Nikhil Iyengar and colleagues (PhysioNet
Fantasia), Andy Schumann and Karl-Jürgen Bär (PhysioNet Autonomic Aging), Tae-Ho Lee, Mara Mather and
colleagues (OpenNeuro ds001242) and Elizabeth DuPre, Wen-Ming Luh and R. Nathan Spreng (OpenNeuro
ds000210).

It is the first step of two workflows: **Cardiovascular Response to Nerve Stimulation**
(`physio-import → stim-events → {ecg-hrv, bp-resp} → stim-dose-response`) and **Heart Rate Variability
and Baroreflex** (`physio-import → {ecg-hrv, bp-resp}`), for human and animal recordings.

## What it does

**For a physiologist.** Give it the files your acquisition system wrote: an EDF from a clinical
system, a LabChart or Spike2 export, a BIOPAC file, an Axon file, a PhysioNet record, a spreadsheet
with a time column, a folder of trials in HDF5, a BIDS dataset with `_physio.tsv.gz` files. It finds
every recording in the folder, decides what each channel is (ECG, arterial pressure, respiration,
pulse, nerve, EMG, stimulation, device heart rate, temperature, other pressure), puts it in volts,
mmHg or beats per minute when the file says which unit it was recorded in (and marks it
*uncalibrated* when the file does not), keeps the markers, comments and annotations, and keeps
stimulation parameters the file stores per trial. You get one figure per recording showing every
channel over the whole recording and in a 10 s zoom, with channels that are flat, clipped or mostly
noise flagged in colour. Check that figure first: it tells you whether the later steps have
something to work with.

**For a data scientist.** One NWB 2.x file per recording (HDF5, gzip + shuffle, chunked); each channel
is a `TimeSeries` in `acquisition` with its own rate, stored as the source's integers with
`conversion`/`offset` where it has them (float sources that are exact multiples of an ADC step are
stored as int16/int32, losslessly); gapped or trial-by-trial data have `timestamps` (shared between
channels on the same clock); markers are a `TimeIntervals` table `markers`, per-trial parameters a
table `stimulation_source`. Index tables `recordings.csv`, `channels.csv` and `events.tsv` describe
everything, so no later step has to open a file to know what is in it. Every file is checked with
NWB Inspector (with PyNWB schema validation). The contract is
`workflow-physiology/docs/conventions.md`.

## Formats

| Format | Reader | Notes |
|---|---|---|
| EDF, EDF+, BDF | pyEDFlib | digital values kept; EDF+ annotations become markers. pyEDFlib rather than Neo's EDF IO: that is itself a wrapper of pyEDFlib, and pyEDFlib gives slice reads and annotations directly |
| WFDB (`.hea` + signal files) | WFDB for Python | every annotation file of the record (`.atr`, `.ecg`, `.qrs`, ...) becomes markers (`annotation`, label = beat code); `Age:`/`Sex:` header comments are read; a unit counts only when the header states it (WFDB assumes mV otherwise, wrong for an uncalibrated channel) |
| Axon ABF (1 and 2), Spike2 `.smr` (32-bit) | Neo | sweeps/blocks become segments; event, marker and TextMark channels become markers. Intan, Blackrock, Plexon `.plx`, BrainVision, Open Ephys binary and TDT are passed to Neo as well (untested here) |
| BIOPAC AcqKnowledge `.acq` | bioread | per-channel rates; event markers |
| NWB | PyNWB | every `TimeSeries` in `acquisition` (2-D series: one channel per column); Subject; interval tables become markers; `stimulation_source` (or `trials`) kept |
| HDF5 | h5py | NWB saved as `.h5`; the "one group per trial, column datasets" layout (datasets with a `_column_names` attribute and a rate attribute such as `frequency_hz`, or a time column), trial-group attributes copied to `stimulation_source` as found, overlapping trial windows cut from one acquisition written once; anything else with a `layout.yaml` |
| MATLAB `.mat` (v4–v7.3) | SciPy / h5py via pymatreader | LabChart "Export > MATLAB" (blocks → segments, comments → markers), Spike2 "Export As > MATLAB" (channel structs; event/marker channels → markers), or a rate (`fs`, `Fs`, `srate`, `sfreq`, `sampling_rate`, ...) or time vector (`t`, `time`, ...) next to equal-length vectors; anything else with a `layout.yaml` |
| CSV, TSV, TXT, XLSX | pandas, openpyxl | a time column (`Time`, `t`, `Time (s)`, `Time (ms)`, ... or `TIME_COLUMN`) or `SAMPLING_RATE`; numeric columns are channels, `ECG (mV)`/`ECG [mV]` give units; empty cells are gaps (NaN, never zero); text columns are markers; LabChart text exports (`Interval=`, `ChannelTitle=`, `#*` comments) are recognised |
| WAV | SciPy | channels `channel 1`, ... uncalibrated; type them with `channels-map.csv` |
| BIDS `_physio.tsv.gz`, `_stim.tsv.gz` | own reader (pandas) | below |
| Spike2 `.smrx` | not read | "export as .smr or CSV from Spike2: the only .smrx reader, CED's sonpy, is closed-source" |
| LabChart `.adicht` | not read | "export as .txt/.mat from LabChart: the only Linux reader is GPL" |

**BIDS physiological recordings** (BIDS specification, "Physiological recordings"): the header-less
`_physio.tsv.gz` / `_stim.tsv.gz` tables with `Columns`, `SamplingFrequency` and `StartTime` from their
JSON sidecars, merged by the inheritance principle (a `task-rest_physio.json` at the dataset root or
a `sub-01_task-rest_physio.json` at subject level applies to every matching file below it; the deeper,
more specific one wins key by key). All `_recording-<label>` files and the `_stim` file of a run become
one recording with each file at its own rate. `StartTime` (seconds relative to the run's first neural
sample, e.g. the first fMRI volume; may be negative) sets each file's start, recording time 0 is the
earliest sample, and the run's `_events.tsv` task events and `_physioevents.tsv.gz` are moved onto the
same timeline, so events stay aligned. `cardiac` is typed ECG when its `Description`/`LongName` or the
recording label says ECG/EKG and PPG (pulse) otherwise, since MR scanners usually record a pulse
oximeter; `respiratory` is respiration; `trigger` in a `_physio` file is the scanner's volume trigger and
is typed `other`, not `stim`, so that stim-events does not mistake volumes for stimulation trains; in a
`_stim` file unrecognised columns are `stim`. `sub`, `ses`, `task`, `acq`, `run` become the recording
id (`sub-01_ses-1_rec-rest-run1`) and `task`/`run` columns of recordings.csv; `participants.tsv`
gives species, sex and age. A dataset that holds only physio files works.

Identical files (`x - Copy.wav`) are read once. Folders named `derivative`, `derivatives`, `code` and
`docs` are not searched (`EXCLUDE_FOLDERS`).

## Ports

| Port | Direction | Contents |
|---|---|---|
| `recordings` | input (subfolder) | recordings in any layout, plus optional `channels-map.csv`, `recordings-map.csv`, `layout.yaml`, SDS `subjects.xlsx`/`performances.xlsx`, BIDS `participants.tsv` |
| `physio` | output | `recordings.csv`, `channels.csv`, `events.tsv`, `provenance.json`, `sub-<subject>/[ses-<session>/]<recording_id>.nwb` |
| `qc` | output | `physio-import/qc.csv`, `nwbinspector.csv`, `figures/<recording_id>_overview.png`, `summary.json`, `provenance.json` |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `SPECIES` | `auto` | species of recordings whose species no mapping file, SDS/BIDS table or NWB file gives (`human`, `pig`, `dog`, `rabbit`, `rat`, `mouse`, `other`); `auto` leaves it empty (later steps assume human) |
| `TIME_COLUMN` | (empty) | time column of tables, in seconds (a unit in the name is honoured); empty = `Time`, `t`, `Time (s)`, `seconds`, `timestamp`, ... |
| `SAMPLING_RATE` | `0` | rate of tables without a time column and of `.mat` vectors without a rate; 0 = skip such files |
| `EXCLUDE_FOLDERS` | `derivative,derivatives,code,docs` | folder names never searched |
| `FIGURES` | `true` | the overview figure per recording |
| `INSPECTOR` | `true` | NWB Inspector on every file |
| `THREADS` | `0` | BLAS threads; 0 = from the container's CPU quota |

This app has no species presets: species only travels to the analysis steps through recordings.csv.

### Mapping files (optional, anywhere in the input)

`channels-map.csv`, the first matching row wins:

```
pattern,type,unit,scale,site,name,path_glob
^channel 1$,ecg,,,lead II,,*.wav            WAV channel 1 is an ECG (stays uncalibrated: no unit given)
ECG,ecg,V,,,,                               the ECG column is in volts
LVP,pressure,mmHg,0.1,left ventricle,lvp,   transducer output x 0.1 = mmHg
```

`pattern` is a case-insensitive regular expression on the channel's original name; `type` one of
`ecg bp resp ppg nerve emg stim hr temp pressure other`; `unit`/`scale`: source value × scale = value
in `unit` (then converted to volts/mmHg/bpm); `name` overrides the NWB name; `path_glob` (an addition
to conventions.md) limits the row to some files.

`recordings-map.csv`: `path_glob, subject, session, label, species` (first match wins).

`layout.yaml`, for HDF5 or `.mat` files no rule above recognises (see `processor/readers/layout.py`):

```yaml
files: "*_ephys_data.mat"     # which files
rate: FsIN                    # a number or the variable holding it (or time: t)
signals:
  - {path: CNAP_avg, name: CNAP average}       # 2-D samples x channels, or trials x samples with `trials`
  - {path: EMG_DuringStim, name: gastric EMG, rate: Fs_EMG}
trials:                       # 2-D signals are trials x samples: one segment per trial
  gap_s: 1                    # back to back when no start times (trials.starts: variable)
  parameters: [Ist, Qst, tst] # per-trial vectors copied to stimulation_source
```

## Outputs

`recordings.csv`: `recording_id, subject, session, label, path, species, start_time, duration_s,
recorded_s, n_segments, channel_types, source_format, source_path, metadata_source, age, sex, task, run`.
Ids are `sub-<subject>[_ses-<session>]_rec-<label>`; subject and session come from
`recordings-map.csv`, then `sub-*`/`ses-*`/`perf-*` folders (or BIDS entities in the file name), then the
file's own metadata, then the file stem; the label is the file stem without what the ids already say,
or the format name (`sub-2_rec-hdf5`). `start_time` is the local wall clock without offset when known;
Unix time stamps (HDF5) are written as UTC.

`channels.csv`: `recording_id, channel` (`ecg`, `ecg-2`, `bp`, ...: the first channel of a type gets
the bare type name), `type, original_name, unit, sampling_rate_hz, n_samples, calibrated, scale, site,
typing` (`mapping`, `name`, `content` or `default`), `status` (`good`, `flat`, `saturated`, `noisy`),
`status_description`.

`events.tsv`: the conventions' columns (`recording_id, onset, duration, trial_type, event_id, ...,
label, source`) plus `annotator` and `marker_channel` when used. `trial_type` is `marker` (file markers,
comments, BIDS task events), `annotation` (EDF+ and WFDB annotations; label = the beat code, e.g. `N`)
or `artifact`: a stretch where a channel flagged `saturated` sits at its minimum or maximum, with
`source` `qc` (an addition to the conventions' source list). Stimulation parameters are added later by
stim-events.

`qc/physio-import/qc.csv`: per channel `n_samples, nan_fraction, min, max, mean, sd, flat_fraction,
saturated_fraction, lag1_autocorrelation, status, storage`; per recording the NWB Inspector message
count.

The checks: **flat** = no change at all, or (not for stimulation and device heart-rate channels) at
least half of the 1 s windows unchanged; **saturated** = at least 1 % of samples at the minimum or
maximum in runs of 30 ms or more (a clipped amplifier or transducer; a slow wave quantised at its peak
does not count); **noisy** = for ECG, BP, respiration, PPG, temperature and pressures a lag-1
autocorrelation below 0.5 (neighbouring samples barely related: mostly noise or mains). These are
import checks, not clinical assessments.

NWB Inspector rates missing subject metadata (age, sex) CRITICAL because DANDI requires it; a file
cannot be given an age it does not have, so these are reported but do not fail the recording. Every
other CRITICAL issue fails the recording.

## Runtime

Docker, amd64 emulation on an M-series Mac, `--cpus 1 --memory 3g`, read-only file system, any uid,
HOME/TMPDIR under /tmp (as Lambda runs it). Peak memory is the container's cgroup `memory.peak`, which
includes the page cache of the files read.

| Demo | Input | Recordings | Wall time | Peak memory | NWB written |
|---|---|---|---|---|---|
| Discover 349 (pig, HDF5) | 496 MB, 20 trials, ECG 1 kHz, HR 500 Hz, 2 nerve cuffs + stimulus line at 30 kHz | 1 | 6.3 s | 740 MB | 21 MB |
| Discover 655 (rat, CSV) | 3 × 20–28 min at 1 kHz, 108 MB | 3 | 4.4 s | 471 MB | 10 MB |
| Discover 406 (human, WAV) | 30 WAV (10 identical copies), 500 Hz | 20 | 10.7 s | 284 MB | 12 MB |
| Fantasia (WFDB) | 6 × 2 h at 250 Hz, 44,167 beat annotations | 6 | 10.3 s | 317 MB | 35 MB |
| Autonomic Aging (WFDB) | 3 × 15–32 min at 1 kHz | 3 | 4.1 s | 290 MB | 17 MB |
| OpenNeuro ds000210 + ds001242 (BIDS) | 3 runs at 50 Hz; 1 run with ECG/resp/SCR at 10 kHz (3.6 M samples) + eye tracker | 4 | 5.8 s | 390 MB | |

Chosen: 1 vCPU, 2048 MiB, 900 s (fits Lambda). Tables and `.mat` files are loaded whole (pandas held
1.7 M rows × 4 columns of Discover 655 in about 470 MB); every other format is read a slice at a time.
Image: 147 MB.

## Validation

- **Synthetic fixture** (`tests/make_fixture.py`; EDF+, WFDB and LabChart-MAT written byte by byte by
  the fixture, not by the libraries the app uses): 42 tests: ids, types, units and scales of every
  channel; R peaks at the planted times; systolic/diastolic within 0.05 mmHg (WFDB) and 1 mmHg (EDF
  16-bit); every WFDB annotation within 1 ms; overlapping HDF5 trials written once; per-trial
  parameters; gaps as NaN; LabChart blocks and comments; BIDS inheritance, `StartTime` and task-event
  alignment; deferred formats; flat and clipped channels flagged; exit codes.
- **Autonomic Aging**: values equal WFDB's physical values (largest difference 2.8e-14 mmHg); record
  0002's NIBP is flagged saturated: three stretches of 47–57 s at 25 or 175 mmHg (the device's limits),
  written as `artifact` events for bp-resp.
- **Discover 349**: 20 trial windows of 25 s that overlap the previous trial by up to 16 s with
  identical samples are written once: 5 contiguous segments, 361.5 s instead of 500 s of windows; every
  value equals the source to float32 precision (relative difference ≤ 3.4e-8) and every timestamp is the
  source's; the nerve channels are stored as int16 at the 0.195 step they were digitised with; the 20
  trials' `current_uA` (50–2500), `frequency_Hz` (2–30), `pulse_width_us`, `duration_s`, `polarity`,
  `stim_shape` are in `stimulation_source`. Species, sex and age from `subjects.xlsx`. ECG and nerve
  channels are uncalibrated (the file gives no units).
- **Discover 655**: start times from `performances.xlsx`; ECG stored as int16 in 1 µV steps; StimAmp
  and CtrlAmp are empty before the controller starts (38–64 % of samples), kept as gaps.
- **Discover 406**: 20 recordings; the 10 `_-_Copy.wav` files skipped as identical; `RD_A.mat` (a
  vector without a rate) and `RD.txt` skipped with the reason; ECG typed from `validation/d406-channels-map.csv`.
- **Fantasia**: all 6 records with age and sex from the header comments; ECG, RESP and BP are
  uncalibrated (no units in the headers; the database describes the BP as uncalibrated).
- **Discover 9** (not a cardiovascular dataset): skipped by default with "layout not recognised";
  with a 10-line `layout.yaml` each animal's 28 averaged nerve responses become 28 segments with
  `Ist`, `Qst`, `tst` per trial.
- **OpenNeuro**: ds000210 sidecars inherited from the subject folder; ds001242's two `recording-` files
  (10 kHz and 60 Hz) of one run in one recording, `cardiac` typed ECG from the recording label
  `EcgRespScr`; task events aligned.
- **Other readers on public sample files**: bioread's test files `physio-4.4.0.acq` and the compressed
  `physio-3.8.1-c.acq` (ECG, respiration, EDA, 10 markers); Neo's test file `File_spike2_1.smr`
  (respiration at 200.8 Hz, 5,711 markers; `File_spike2_2.smr` holds only spikes and is skipped).
- Every NWB file written passes NWB Inspector without CRITICAL issues other than missing subject age.

## Limitations

- It does not judge the physiology: a channel typed `ecg` may still be too noisy for beat detection
  (ecg-hrv decides), and the checks only see flat, clipped and noise-like signals.
- Units are only as good as the file: no unit means `calibrated=false` (relative amplitudes only),
  even when you know the value is in mV; say so in `channels-map.csv`.
- Typing by name can be wrong for unusual names; check `typing` in channels.csv and override.
- `.smrx` and `.adicht` are not read (licences); EDF+D (discontinuous) is read as continuous;
  multi-segment WFDB records and FLAC-coded WFDB signals are not supported; Neo formats other than ABF
  and SMR are untested here.
- Tables and MATLAB files are loaded whole: a several-hour, high-rate CSV needs more memory than 2 GB.
- Anaesthesia, species and recording setup are not visible to this step; they matter for every later one.
- The figures and checks are research outputs, not a clinical assessment of the recordings.

## Licences

Apache-2.0 (this code). Bundled components and their licences: `THIRD_PARTY.md`. Demo data: Discover
datasets CC BY 4.0, Fantasia ODC-By 1.0, Autonomic Aging ODbL 1.0, OpenNeuro ds000210 and ds001242
PDDL 1.0.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in CITATION.cff.

- Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K, Frank L,
  Bouchard KE (2022). The Neurodata Without Borders ecosystem for neurophysiological data science.
  *eLife* 11:e78362. doi:10.7554/eLife.78362, PMID 36193886.
- Garcia S, Guarino D, Jaillet F, Jennings T, Pröpper R, Rautenberg PL, Rodgers CC, Sobolev A,
  Wachtler T, Yger P, Davison AP (2014). Neo: an object model for handling electrophysiology data in
  multiple formats. *Frontiers in Neuroinformatics* 8:10. doi:10.3389/fninf.2014.00010, PMID 24600386.
- Xie C, McCullum L, Johnson A, Pollard T, Gow B, Moody B (2023). Waveform Database Software Package
  (WFDB) for Python. PhysioNet. doi:10.13026/9njx-6322.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K,
  Stanley HE (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for
  Complex Physiologic Signals. *Circulation* 101(23):e215-e220. doi:10.1161/01.CIR.101.23.e215, PMID 10851218.
- Vack N, Ollinger J, Fitch D, Davison M (2020). uwmadison-chm/bioread. Zenodo.
  doi:10.5281/zenodo.3374011.
- pyEDFlib (Holger Nahrstaedt and contributors), https://github.com/holgern/pyedflib; pymatreader
  (Dirk Gütlin, Thomas Hartmann), https://gitlab.com/obob/pymatreader; h5py (Andrew Collette and
  contributors), https://github.com/h5py/h5py; NWB Inspector, https://github.com/NeurodataWithoutBorders/nwbinspector.
- Harris CR et al. (2020). Array programming with NumPy. *Nature* 585:357-362.
  doi:10.1038/s41586-020-2649-2, PMID 32939066.
- Virtanen P et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python.
  *Nature Methods* 17:261-272. doi:10.1038/s41592-019-0686-2, PMID 32015543.
- McKinney W (2010). Data Structures for Statistical Computing in Python. *Proceedings of the Python
  in Science Conference* 56-61. doi:10.25080/Majora-92bf1922-00a; The pandas development team, pandas,
  Zenodo, doi:10.5281/zenodo.3509134.
- Hunter JD (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*
  9:90-95. doi:10.1109/MCSE.2007.55.
- BIDS contributors. The Brain Imaging Data Structure (BIDS) Specification. Zenodo.
  doi:10.5281/zenodo.3686061; Gorgolewski KJ et al. (2016). The brain imaging data structure, a format
  for organizing and describing outputs of neuroimaging experiments. *Scientific Data* 3:160044.
  doi:10.1038/sdata.2016.44, PMID 27326542.
- Bandrowski A et al. (2021). SPARC Data Structure: Rationale and Design of a FAIR Standard for
  Biomedical Research Data. bioRxiv. doi:10.1101/2021.02.10.430563.

Demo data:

- Armitage O et al. (2023). Influence of vagus nerve stimulation on cardiac activity in pigs.
  Pennsieve Discover. doi:10.26275/dm5l-tspb.
- Bender SA et al. (2026). Single-electrode bidirectional control of heart rate via vagus nerve
  modulation in rat model. Pennsieve Discover. doi:10.26275/5cwo-mkpz.
- Ali MK et al. (2024). Best parameters of heart rate variability for assessing autonomic responses to
  brief rectal distention in patients with irritable bowel syndrome. Pennsieve Discover.
  doi:10.26275/nung-b9bm.
- Ward M et al. (2020). Influence of left vagal stimulus pulse parameters on vagal and gastric activity
  in rat. Pennsieve Discover. doi:10.26275/qh3q-elj6.
- Iyengar N et al. (1994). Fantasia Database. PhysioNet. doi:10.13026/C2RG61; Iyengar N et al. (1996).
  Age-related alterations in the fractal scaling of cardiac interbeat interval dynamics. *Am J Physiol
  Regul Integr Comp Physiol* 271:R1078-R1084. doi:10.1152/ajpregu.1996.271.4.R1078, PMID 8898003.
- Schumann A, Bär K (2021). Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic
  function during healthy aging. PhysioNet. doi:10.13026/2hsy-t491; Schumann A, Bär K-J (2022). Autonomic
  aging – A dataset to quantify changes of cardiovascular autonomic function during healthy aging.
  *Scientific Data* 9:95. doi:10.1038/s41597-022-01202-y, PMID 35322044. (Cite Goldberger et al. 2000
  with every PhysioNet dataset.)
- Lee T-H, Greening S, Ueno T, Clewett D, Ponzio A, Sakaki M, Mather M (2019). Examining effects of
  arousal on responses to salient and non-salient stimuli in younger and older adults. OpenNeuro.
  doi:10.18112/openneuro.ds001242.v1.0.0.
- DuPre E, Luh W-M, Spreng RN. Multi-echo fMRI replication sample of autobiographical memory,
  prospection and theory of mind reasoning tasks. OpenNeuro ds000210, snapshot 00002,
  https://openneuro.org/datasets/ds000210/versions/00002.

## Acknowledgements

Neo is developed by the NeuralEnsemble community; WFDB and PhysioNet by the MIT Laboratory for
Computational Physiology; NWB, PyNWB and NWB Inspector by the Neurodata Without Borders project; bioread by the University of Wisconsin-Madison Center for Healthy Minds. The demo
datasets on Pennsieve Discover were funded by the NIH SPARC program; please acknowledge the funders the
datasets' own descriptions name.

## Development

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python        # 42 tests on the synthetic fixture
make run PYTHON=.venv/bin/python         # Docker over the fixture into data/output/
make lambda-check PYTHON=.venv/bin/python
validation/run-demos.sh                  # the demo data (DOCKER=1 to use the image); logs in validation/logs/
```
