# Sample input

`make fixture` writes a synthetic input (`data/fixture/recordings/`, every supported format, with known
contents in `data/fixture/truth.json`). Real data used to validate the app:

| Source | What to fetch | Notes |
|---|---|---|
| Pennsieve Discover 349 (doi:10.26275/dm5l-tspb, CC BY 4.0) | `files/primary/sub-2/subject-2.hdf5` (496 MB), `files/subjects.xlsx` | pig VNS grid; one HDF5 group per stimulation trial |
| Pennsieve Discover 655 (doi:10.26275/5cwo-mkpz, CC BY 4.0) | `files/primary/sub-*/perf-*/*.smrx.csv`, `files/subjects.xlsx`, `files/performances.xlsx` | rat; the CSV exports of the Spike2 `.smrx` files. Add `validation/d655-channels-map.csv` (ECG in volts) |
| Pennsieve Discover 406 (doi:10.26275/nung-b9bm, CC BY 4.0) | `files/primary/sub-*/perf-*/*.wav`, `files/subjects.xlsx` | human single-lead ECG as WAV. Add `validation/d406-channels-map.csv` (WAV channel 1 = ECG) |
| PhysioNet Fantasia (doi:10.13026/C2RG61, ODC-By 1.0) | `f1o01`, `f1y01`, `f2o01`, `f2o02`, `f2y01`, `f2y02` `.hea/.dat/.ecg` | WFDB with reference beat annotations; set SPECIES=human |
| PhysioNet Autonomic Aging (doi:10.13026/2hsy-t491, ODbL 1.0) | `0001`-`0003` `.hea/.dat` | calibrated NIBP; set SPECIES=human |
| OpenNeuro ds001242 v1.0.0 (doi:10.18112/openneuro.ds001242.v1.0.0, PDDL) and ds000210 (PDDL) | `validation/fetch-openneuro.sh` | BIDS `_physio.tsv.gz` recordings |

`scripts/fetch-sample-data.sh` in the workflow-physiology repository stages the five demo sources under its
`data/`; `validation/run-demos.sh` runs this app on each of them (natively, or in Docker with `DOCKER=1`).
