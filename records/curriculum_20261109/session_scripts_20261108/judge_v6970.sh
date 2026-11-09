#!/bin/zsh
# Fetch the completed native v69/v70 arms from Spark and judge them exactly as the local
# gb1/od1 arms were judged: full-96 flat (--skip-terrain) + omni15 at u600/u1200/u2000,
# unchanged gates. Verifies each declaration names the fetched bytes before judging.
set -o pipefail
CUPI=/Users/new/Desktop/cupi/hexapod-cupi
C=/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum
R=/srv/cupi/hexapod/runs/julian
H=julian@spark-e26c.tailf4bbf2.ts.net
cd $CUPI || exit 1

typeset -A RUNS
RUNS=(nv69 dest_v69_ghost_critic_u2000_seed20261009_20261107_001
      nv70 dest_v70_odom_u2000_seed20261011_20261107_001)

for TAG in nv69 nv70; do
  NAME=${RUNS[$TAG]}
  for U in 000600 001200 002000; do
    SHORT=$((10#$U))
    DEST=$C/native_chain/${TAG}_u${SHORT}
    [[ -f $DEST/checkpoint_update$U.pt ]] && continue
    mkdir -p $DEST
    echo "=== fetching $TAG u$SHORT ($NAME) ==="
    scp -o BatchMode=yes $H:$R/$NAME/run/standing/checkpoint_update$U.pt $DEST/ || { echo "SCP FAILED $TAG u$SHORT"; exit 1; }
    scp -o BatchMode=yes $H:$R/$NAME/run/standing/checkpoint_update$U.json $DEST/ || { echo "SCP FAILED $TAG u$SHORT declaration"; exit 1; }
    SHA=$(shasum -a 256 $DEST/checkpoint_update$U.pt | cut -d" " -f1)
    DECL=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_sha256'])" $DEST/checkpoint_update$U.json)
    echo "  checkpoint sha256 $SHA"
    echo "  declaration sha256 $(shasum -a 256 $DEST/checkpoint_update$U.json | cut -d' ' -f1)"
    if [[ $SHA != $DECL ]]; then echo "DECLARATION MISMATCH $TAG u$SHORT: $SHA != $DECL"; exit 1; fi
  done
done

judge_arm() {
  local TAG=$1 CKPT=$2
  echo "=== judging $TAG (flat only) ==="
  uv run python $C/eval_checkpoint.py $TAG $CKPT --skip-terrain || { echo "EVAL FAILED $TAG"; return 1; }
  uv run python $C/omni15_suite.py --checkpoint $CKPT --output $C/omni15_$TAG > $C/omni15_$TAG.log 2>&1 \
    || { echo "OMNI FAILED $TAG"; tail -5 $C/omni15_$TAG.log; return 1; }
  uv run python - <<PY
import json
cases = json.load(open("$C/omni15_$TAG/surrogate_summary.json"))["cases"]
omni = [c for c in cases if "omni15" in c["case_id"]]
print("omni15 $TAG:", sum(c["pass"] for c in omni), "/", len(omni))
PY
}

for TAG in nv69 nv70; do
  for SHORT in 600 1200 2000; do
    judge_arm ${TAG}_u${SHORT} $C/native_chain/${TAG}_u${SHORT}/checkpoint_update$(printf %06d $SHORT).pt
  done
done
echo "=== native v69/v70 judging complete ==="
