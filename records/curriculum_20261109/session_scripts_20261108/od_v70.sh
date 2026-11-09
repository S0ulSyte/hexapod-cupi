#!/bin/zsh
# Odometry-ghost round: destination_v70 (v59 reward, actor ghost from leg odometry, critic privileged) warm-started from tw_s3_u600,
# critic keeps the privileged set) from the narrowed tw_s3_u600 start, 2000 updates,
# judged flat-only (full-96 + omni15, terrain skipped at Julian's instruction).
set -o pipefail
CUPI=/Users/new/Desktop/cupi/hexapod-cupi
C=/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum
cd $CUPI || exit 1

echo "=== odometry arm od1 (destination_v70, u2000, anchor 0.006, seed 20261011, init tw_s3/checkpoint_update000600.pt) ==="
uv run python -m locomotion.surrogate.train \
  --output $C/od1 \
  --task "locomotion.omniwalk.destination_task:experiment_task(destination_v70)" \
  --wrapper locomotion.rl.destination_wrapper:DestinationVecEnv \
  --initialize-from $C/tw_s3/checkpoint_update000600.pt --initialize-rate checkpoint \
  --action-mean tanh --observation-normalization none --observation-scaling fixed \
  --command-segments bootstrap --learning-rate-max 1e-4 --action-std 0.05 --action-std-final 0.05 \
  --gait-clock 60 --action-smoothing mean2 --velocity-noise 0.5 --episode-seconds 32.0 \
  --updates 2000 --seed 20261011 --threads 6 --anchor-weight 0.006 \
  --ppo-config-overrides '{"num_steps_per_env": 36, "algorithm": {"num_mini_batches": 6}, "environment_wrapper": {"destination_input": "terrain_v2_odom", "gait_clock_scaling": "speed_v1"}}' \
  > $C/od1_train.log 2>&1
RC=$?
tail -2 $C/od1_train.log
if [[ $RC -ne 0 ]]; then echo "TRAIN FAILED od1 rc=$RC"; exit 1; fi

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

judge_arm od1_u600 $C/od1/checkpoint_update000600.pt
judge_arm od1_u1200 $C/od1/checkpoint_update001200.pt
judge_arm od1_u2000 $C/od1/checkpoint_update002000.pt

echo "=== ghost-blind round complete ==="
