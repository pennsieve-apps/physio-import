#!/bin/sh
# A few BIDS physiological recordings from two public OpenNeuro datasets (anonymous S3, no account), used to
# check the BIDS physio reader on real data. Written to data/openneuro/ (gitignored).
#   ds000210 snapshot 00002 (PDDL; DuPre, Luh, Spreng): pulse and respiration at 50 Hz, sidecars at subject level
#   ds001242 v1.0.0 (PDDL; doi:10.18112/openneuro.ds001242.v1.0.0; Lee et al.): ECG, respiration, SCR, trigger,
#            marker at 10 kHz and an eye tracker at 60 Hz, two recording-<label> files in one run
set -eu
OUT=$(cd "$(dirname "$0")/.." && pwd)/data/openneuro
get() { aws s3 cp --no-sign-request --only-show-errors "s3://openneuro.org/$1/$2" "$OUT/$1/$2"; }
for f in dataset_description.json participants.tsv README CHANGES sub-01/sub-01_task-cuedSGT_physio.json \
         sub-01/sub-01_task-rest_physio.json sub-01/func/sub-01_task-cuedSGT_run-01_physio.tsv.gz \
         sub-01/func/sub-01_task-cuedSGT_run-01_events.tsv sub-01/func/sub-01_task-cuedSGT_run-02_physio.tsv.gz \
         sub-01/func/sub-01_task-cuedSGT_run-02_events.tsv sub-01/func/sub-01_task-rest_run-01_physio.tsv.gz; do
  get ds000210 "$f"
done
for f in dataset_description.json participants.tsv README CHANGES sub-01/func/sub-01_task-fearlearning_events.tsv \
         sub-01/func/sub-01_task-fearlearning_recording-EcgRespScr_physio.json \
         sub-01/func/sub-01_task-fearlearning_recording-EcgRespScr_physio.tsv.gz \
         sub-01/func/sub-01_task-fearlearning_recording-eye_physio.json \
         sub-01/func/sub-01_task-fearlearning_recording-eye_physio.tsv.gz; do
  get ds001242 "$f"
done
echo "run: INPUT_DIR=$OUT OUTPUT_DIR=data/output-openneuro SPECIES=human python -m processor.main"
