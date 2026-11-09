#!/bin/bash
# Native replay of the two sim2real input arms from the chain3 model (nc3 u600, f62e4ce3...):
# stage A = destination_v69 (actor without the nine ghost inputs, critic privileged) from the
# narrowed nc3 start (036a6432...), stage B = destination_v70 (actor ghost from leg odometry)
# from nc3 verbatim. u2000 each, anchor 0.006, seeds 20261009/20261011, flat, no terrain —
# the native mirror of the local gb1/od1 runs. Freeze b669acdae5c3 (HEAD cf9f7318).
# Queue etiquette: launch only at GPU <= 25%; a lost flock race re-queues instead of failing.
R=/srv/cupi/hexapod/runs/julian
OUT=/home/julian/v6970.log
MARK=/home/julian/v6970.done
SA=dest_v69_ghost_critic_u2000_seed20261009_20261107_001
SB=dest_v70_odom_u2000_seed20261011_20261107_001
rm -f "$MARK"
{
run_stage() {
  local N=$1 FINAL=$2
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
run_stage $SA checkpoint_update002000.pt
run_stage $SB checkpoint_update002000.pt
echo "status: finished" > "$MARK"
} >> "$OUT" 2>&1
