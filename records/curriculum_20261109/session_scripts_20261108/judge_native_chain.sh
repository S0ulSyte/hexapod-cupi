#!/bin/zsh
# Copy the completed native v59 spine chain down from Spark, verify the recorded hashes,
# and judge each stage on the full surrogate suites (full-96 + terrain + omni15).
set -o pipefail
CUPI=/Users/new/Desktop/cupi/hexapod-cupi
C=/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum
R=/srv/cupi/hexapod/runs/julian
H=julian@spark-e26c.tailf4bbf2.ts.net
cd $CUPI || exit 1

typeset -A RUNS UPDATES PREFIX
RUNS=(nc1 dest_v59_chain1_u1400_seed20260917_20261107_001
      nc2 dest_v59_chain2_u600_seed20260922_20261107_001
      nc3 dest_v59_chain3_u600_seed20260924_20261107_001)
UPDATES=(nc1 001400 nc2 000600 nc3 000600)
PREFIX=(nc1 7e49b9ebdede nc2 5c3f71d45ec9 nc3 f62e4ce330eb)

for TAG in nc1 nc2 nc3; do
  NAME=${RUNS[$TAG]} U=${UPDATES[$TAG]}
  DEST=$C/native_chain/${TAG}_u${U#00}
  mkdir -p $DEST
  echo "=== fetching $TAG ($NAME u$U) ==="
  scp -o BatchMode=yes $H:$R/$NAME/run/standing/checkpoint_update$U.pt $DEST/ || { echo "SCP FAILED $TAG"; exit 1; }
  scp -o BatchMode=yes $H:$R/$NAME/run/standing/checkpoint_update$U.json $DEST/ || { echo "SCP FAILED $TAG declaration"; exit 1; }
  SHA=$(shasum -a 256 $DEST/checkpoint_update$U.pt | cut -d" " -f1)
  DSHA=$(shasum -a 256 $DEST/checkpoint_update$U.json | cut -d" " -f1)
  echo "  checkpoint sha256 $SHA"
  echo "  declaration sha256 $DSHA"
  if [[ $SHA != ${PREFIX[$TAG]}* ]]; then echo "HASH MISMATCH $TAG: $SHA !~ ${PREFIX[$TAG]}"; exit 1; fi
done

judge() {
  local TAG=$1 CKPT=$2
  echo "=== judging $TAG ==="
  uv run python $C/eval_checkpoint.py $TAG $CKPT || { echo "EVAL FAILED $TAG"; return 1; }
  uv run python $C/omni15_suite.py --checkpoint $CKPT --output $C/omni15_$TAG > $C/omni15_$TAG.log 2>&1 \
    || { echo "OMNI FAILED $TAG"; tail -5 $C/omni15_$TAG.log; return 1; }
  uv run python - <<PY
import json
cases = json.load(open("$C/omni15_$TAG/surrogate_summary.json"))["cases"]
omni = [c for c in cases if "omni15" in c["case_id"]]
print("omni15 $TAG:", sum(c["pass"] for c in omni), "/", len(omni))
PY
}

judge nc1_u1400 $C/native_chain/nc1_u1400/checkpoint_update001400.pt
judge nc2_u600 $C/native_chain/nc2_u600/checkpoint_update000600.pt
judge nc3_u600 $C/native_chain/nc3_u600/checkpoint_update000600.pt

echo "=== native chain judging complete ==="
