#!/bin/sh
# Run physio-import on the demo data of the physiology workflows (natively, or in Docker with DOCKER=1) and
# write each demo's output to $WORK/<demo>/physio-import/ (physio/ and qc/), where the later apps read it.
# Each demo's input folder is a folder of symlinks: the staged data plus the mapping files of this folder.
#   validation/run-demos.sh [demo ...]      demos: d349 d655 d406 fantasia aging (default: all)
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${DATA:-$HERE/../workflow-physiology/data}" && pwd -P)
WORK=${WORK:-$DATA/work}
PYTHON=${PYTHON:-python3}
IMAGE=${IMAGE:-physio-import:dev}
DEMOS=${*:-d349 d655 d406 fantasia aging}
for demo in $DEMOS; do
  species=""
  case $demo in
    d349) src=discover-349 ;;
    d655) src=discover-655 ;;
    d406) src=discover-406 ;;
    fantasia) src=fantasia; species=human ;;
    aging) src=autonomic-aging; species=human ;;
    *) echo "unknown demo $demo" >&2; exit 2 ;;
  esac
  in=$HERE/data/demo-input/$demo
  out=$WORK/$demo/physio-import
  mkdir -p "$in"
  ln -sfn "$DATA/$src" "$in/$src"
  [ -f "$HERE/validation/$demo-channels-map.csv" ] && cp "$HERE/validation/$demo-channels-map.csv" "$in/channels-map.csv"
  rm -rf "$out/physio" "$out/qc"
  mkdir -p "$out" && chmod 777 "$out"
  echo "=== $demo: $DATA/$src -> $out"
  if [ "${DOCKER:-0}" = 1 ]; then
    docker run --rm --platform linux/amd64 --cpus 1 --memory "${MEMORY:-3g}" --read-only --user 65534:65534 \
      --tmpfs /tmp:mode=1777 -e HOME=/tmp/w/home -e TMPDIR=/tmp/w/tmp -e INPUT_DIR=/tmp/s/input \
      -e OUTPUT_DIR=/tmp/s/output -e SPECIES="${species:-auto}" \
      -v "$in":/tmp/s/input:ro -v "$DATA/$src":"$DATA/$src":ro -v "$out":/tmp/s/output "$IMAGE"
  else
    (cd "$HERE" && INPUT_DIR="$in" OUTPUT_DIR="$out" SPECIES="${species:-auto}" "$PYTHON" -m processor.main)
  fi
done
