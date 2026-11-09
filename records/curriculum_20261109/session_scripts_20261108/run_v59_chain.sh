#!/bin/bash
# v59 spine chain: the three anchored runs that produced the v59 champion, replayed natively
# in sequence at commit 6caef53e (freeze 0c021250...): chain1 = destination_v59 u1400 from the
# widened v29_terrain_v2 start (seed 20260917), chain2 = u600 from chain1 u1400 (seed 20260922),
# chain3 = u600 from chain2 u600 (seed 20260924); anchor 0.006 throughout, flat, no terrain.
# Queue etiquette: launch only at GPU <= 25%; a lost flock race re-queues instead of failing.
# Stages 2 and 3 ship placeholder init hashes; this runner fills the real ones from the
# previous stage's output before the pack approval, which refreezes the binding.
R=/srv/cupi/hexapod/runs/julian
OUT=/home/julian/v59_chain.log
MARK=/home/julian/v59_chain.done
S1=dest_v59_chain1_u1400_seed20260917_20261107_001
S2=dest_v59_chain2_u600_seed20260922_20261107_001
S3=dest_v59_chain3_u600_seed20260924_20261107_001
rm -f "$MARK"
{
run_stage() {
  local N=$1 PREV=$2 FINAL=$3
  for attempt in 1 2 3 4 5 6; do
    for i in $(seq 1 288); do
      U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
      echo "queue[$N][$attempt]: check $i gpu ${U}%"
      [ "$U" -le 25 ] && break
      sleep 300
    done
    U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$U" -gt 25 ]; then
      echo "gpu still ${U}% after queue window $attempt of $N; re-queueing"
      continue
    fi
    if [ ! -d "$R/$N/source" ]; then
      cd $R && python3 -m tarfile -e /home/julian/$N.tar . 2>/dev/null || true
    fi
    if [ -n "$PREV" ]; then
      python3 - "$R/$N/binding.json" "$PREV" <<'PYEOF'
import hashlib
import json
import sys

binding_path, ckpt = sys.argv[1], sys.argv[2]
decl = ckpt[:-3] + ".json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


cs, ds = sha(ckpt), sha(decl)
b = json.load(open(binding_path))
b["input_files"][ckpt] = cs
b["input_files"][decl] = ds
a = b["command_args"]
a[a.index("--initialize-from-sha256") + 1] = cs
a[a.index("--initialize-from-declaration-sha256") + 1] = ds
with open(binding_path, "w") as f:
    json.dump(b, f, indent=1)
print(f"patched init hashes: {ckpt} {cs[:12]} / declaration {ds[:12]}")
PYEOF
      if [ $? -ne 0 ]; then
        echo "status: failed:${N}:init_patch" > "$MARK"
        exit 0
      fi
    fi
    cd $R/$N/source && python3 -m locomotion.spark.pack approve --output $R/$N | grep -E "binding_sha256" || true
    setfacl -R -m u:1000:rX $R/$N && setfacl -R -m d:u:1000:rwX $R/$N
    B=$(sha256sum $R/$N/binding.json | cut -d" " -f1); echo "binding $B"
    cd $R/$N/source && PYTHONDONTWRITEBYTECODE=1 timeout -s KILL 22500 python3 -m locomotion.spark.launch --bindings ../binding.json --bindings-sha256 $B > /home/julian/${N}_launch_full.log 2>&1
    X=$?
    echo "launch exit: $X ($N attempt $attempt)"
    if [ "$X" -eq 0 ] && [ -f "$R/$N/run/standing/$FINAL" ]; then
      echo "stage $N complete: $(sha256sum $R/$N/run/standing/$FINAL)"
      L=$(ls $R/$N/run/logs/*.log 2>/dev/null | tail -1)
      grep -E "Iteration time|Mean reward|Mean episode length" "$L" 2>/dev/null | tail -6
      return 0
    fi
    if grep -q "BlockingIOError" /home/julian/${N}_launch_full.log; then
      echo "lost the gpu lock race to another workload; back to the queue"
      sleep 300
      continue
    fi
    grep -iE "error|refus|differ|abort|traceback|fatal|egmentation|KeyError" /home/julian/${N}_launch_full.log | head -20 || true
    echo "status: failed:${N}" > "$MARK"
    exit 0
  done
  echo "status: queued_out:${N}" > "$MARK"
  exit 0
}
run_stage $S1 "" checkpoint_update001400.pt
run_stage $S2 "$R/$S1/run/standing/checkpoint_update001400.pt" checkpoint_update000600.pt
run_stage $S3 "$R/$S2/run/standing/checkpoint_update000600.pt" checkpoint_update000600.pt
echo "status: finished" > "$MARK"
} >> "$OUT" 2>&1
