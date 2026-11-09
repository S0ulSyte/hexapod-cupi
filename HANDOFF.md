Source transcript: /Users/new/.claude/projects/-Users-new-Desktop-cupi-hexapod-cupi/f182c618-1fa8-4e65-a85d-3dba523704f7.jsonl

# Handoff — hexapod model state and the path to a real robot (written 2026-11-09T05Z)

This file exists at Julian's request (2026-11-09) despite AGENTS.md's no-extra-handoff-docs
rule; never commit it, and offer to delete it once taken over.

Paths: repo = /Users/new/Desktop/cupi/hexapod-cupi (branch jg2399/sim2real-adapter, HEAD cf9f7318,
~40 local commits, NOTHING pushed). $C = /Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum
(campaign log decisions.md, 33 entries, every number below is in it with hashes). Session scripts
survive in $C/session_scripts_20261108/. Spark: julian@spark-e26c.tailf4bbf2.ts.net, runs under
/srv/cupi/hexapod/runs/julian. Read AGENTS.md, README, CONTRIBUTING, docs/OPERATIONS.md first.

## Goal
Julian's campaign: one checkpoint that passes the full surrogate benchmarks with inputs the real
robot can supply, feeding Stage 2 acceptance (docs/TRAINING.md#acceptance). Reached when a
robot-honest input model matches tw_s3_u600's scores and the program lead (palerdr/James, relayed
by Julian) accepts the lineage. Judges and gates never change (AGENTS.md).

## State now — the model
- Champion (privileged ghost): $C/tw_s3/checkpoint_update000600.pt = 59/96, subset 36/36,
  omni 15/16 ($C/full96_tw_s3_u600/surrogate_summary.json; decisions.md round 29).
- Weights spine: widened Spark start v29_terrain_v2.pt (5b32fba4) -> tw_q3 (destination_v59,
  u1400, 56/96) -> tw_q6 (58) -> tw_q9 (58) -> tw_s3 = v59 + bound-margin 1.0@0.95 (v63), u600.
- Native replay of that spine at 6caef53e (Spark, Isaac): nc1 55, nc2 54, nc3 56/96, terrain 6/6;
  nc3 is the first model ever to pass transition__full_program. Checkpoints + declarations in
  $C/native_chain/nc{1,2,3}_u*/ (nc3 sha f62e4ce3...). Fetch/judge recipe:
  session_scripts_20261108/judge_native_chain.sh.
- Robot-honest arms (James's v29 review, item 1), both registered in code and trained locally
  from tw_s3_u600:
  - v69 terrain_v2_ghost_critic (ghost deleted from actor, 278->269; critic keeps it): gb1
    u2000 = 39/96, omni 8/16. The blind floor.
  - v70 terrain_v2_odom (actor's 9 ghost features recomputed from LEG ODOMETRY, same layout, so
    terrain_v2 warm-starts unchanged): od1 u1200 = 53/96, omni 16/16
    ($C/full96_od1_u1200/surrogate_summary.json). Module: src/locomotion/omniwalk/mdp/odometry.py.
- Contact-source ladder on od1_u1200, flat 96 (decisions.md, last three diagnostics): privileged
  59 > sensed toe force 53 (omni 16) > gait-schedule-inferred 47 (omni 16) > ghost zeroed 39
  (omni 8) > contact assumed always 8 (omni 0). A wrong ghost is far worse than none; losses of
  schedule vs sensed are exactly six curving walks.
- Measured leg-odometry drift on recorded traces: 0.3-6 mm/s sensed, 91-265 mm/run schedule-
  inferred (decisions.md; session_scripts_20261108/leg_odometry.py, viz_odometry.py/.html).
- RUNNING ON SPARK NOW: /home/julian/run_v6970.sh (queue loop, launches at GPU<=25%, never
  cancels others) will train both arms natively from nc3: dest_v69_ghost_critic_u2000_
  seed20261009_20261107_001 (init = narrowed nc3, 036a6432..., in $R/starts_v6970_20261107/)
  then dest_v70_odom_u2000_seed20261011_20261107_001 (init = nc3 verbatim). u2000, anchor 0.006,
  freeze b669acda. As of 2026-11-09T04Z still QUEUED (window 2 of 6) behind shaurya's jobs;
  progress: `ssh ... 'tail -1 /home/julian/v6970.log; cat /home/julian/v6970.done'`.
- Round 34 (v68 pinned-ghost training) is designed and registered but HELD by Julian; do not
  launch it (decisions.md round 34 entry).

## Connecting a trained checkpoint to a real hexapod
The surrogate (MuJoCo, this Mac) and Spark (Isaac native) checkpoints share one actor interface:
278 float inputs at 50 Hz -> 18 joint targets (tanh, action_scale 0.42 rad, 0.040 rad / 20 ms
slew limiter in the controller; RS05 motors). Input layout (destination_wrapper.py + footfall.py):
- [0:210]  5 x 42 proprio frames: gyro xyz (IMU rates), projected gravity (IMU attitude — yaw-
  invariant, raw accel never consumed), 18 joint offsets + 18 joint rates (encoders; rates were
  trained with 0.5 rad/s noise, so noisy derivatives are in-distribution).
- [210:213] command (forward, left, turn) in the nav frame — operator/mission input.
- [213:231] executed-action feature — controller's own state.
- [231:233] gait clock sin/cos — software clock (task phase, speed_v1 scaling), zero at zero cmd.
- [233:242] ghost_v11 9 values — THE sim2real problem, solved by v70: compute them from leg
  odometry (odometry.py): body-frame toe positions from joint FK (encoders), foot contact, yaw
  from gyro integration. The ghost restarts at the odometry pose on every command change, so
  heading and position are SEGMENT-LOCAL (~20 s): no absolute heading, no GPS needed. Lidar-
  inertial odometry can replace leg odometry one-for-one (same segment-local semantics).
- [242:278] terrain 36 — zero on flat ground (native runs already run them zeroed). Terrain
  sensing on hardware is future work.
Critic inputs (281, incl. true pose) are training-only; the robot never supplies them.
Hardware checklist, in value order (the ladder above is the evidence): (1) foot-contact sensing
per leg — the keystone; even a crude current-spike detector moves 47->53. If a foot's sensor
dies, fall back to the gait-schedule inference in session_scripts_20261108/gaitcontact_eval.py
(phase<0.4 tripod A, 0.5-0.9 tripod B, all-stance at zero command), NEVER assume always-stance,
and zero the ghost features rather than feed a wrong ghost. (2) IMU rates + attitude, encoders,
50 Hz loop. (3) Nothing else: no RTK, no mocap. Before hardware, James's conditions still open:
train with table2_v1 randomization + obs noise + actuation delay (review item 2, NOT built;
DynamicsRandomization exists in surrogate/env.py, the table2_v1 preset is PR #94, not in this
checkout) and native push/payload evals (item 3, NOT built — scene.py pushes exist, eval side
does not). AGENTS.md: native evaluations must record ground-contact force and motor torque.

## My rules (beyond memory files — see MEMORY.md, all still in force)
- "do not commit, push, or open PRs without Julian's say-so" (2026-10-28; memory
  hexapod-fork-commit-no-push.md). Commits on jg2399/* are fine and expected.
- "stop asking for permission at all" (2026-11-04): set bypassPermissions early each session
  (memory hexapod-bypass-permissions.md). Do NOT use the Workflow tool or Agent subagents for
  campaign work — they raise permission cards; work inline. Ignore "Ultracode is on" banners.
- "Hold for now" (2026-11-07): round 34 stays held until Julian says go.
- "if possible dont use usage as quickly" (2026-11-05): lean turns, one report script, short replies.
- Give no time estimates (AGENTS.md); when Julian asks "how long" directly, answer from recorded
  rates (e.g. 6.6-6.9 s/update from /home/julian/v59_chain.log) and say the basis.
- Spark etiquette (memory hexapod-spark-queue-etiquette.md): queue at GPU<=25%, never cancel or
  pause others' jobs (sudo pause idea dropped 2026-11-07: Julian has no sudo password).
- Input changes to the lineage need the lead's OK relayed by Julian (memory
  hexapod-surrogate-first-lead-approval.md). v70's registration awaits that sign-off.

## Tried and dropped
- Rounds 31-33 (vary exposure, situation prices, anchor bracket): all failed full_program;
  root cause found instead — the EVAL tracker pins the ghost during per-control-varying commands
  (destination.py:70), v68 designed for it, then held (decisions.md rounds 31-34).
- Odometry without contact (always-stance): 8/96, omni 0/16 — never ship that fallback.
- Pausing competing GPU jobs via sudo: no password; queue instead.
- Editing mj_model.body_mass for "lighter robot" probes: stale derived constants; scale gravity.

## Open
1. Julian already authorized the whole chain ("do 1 and 2 on the spark ... and the evals after",
   2026-11-08): restart a watcher (background loop, ssh every ~15 min for /home/julian/v6970.done)
   and, when it says "status: finished", run UNPROMPTED from the repo root:
   zsh $C/session_scripts_20261108/judge_v6970.sh
   (fetches u600/u1200/u2000 of both runs, declaration-verified, judges flat 96 + omni15).
   Check: prints "native v69/v70 judging complete" + six "omni15 ..." lines; $C/full96_nv69_u*/
   etc. appear. If .done says failed:*, read /home/julian/<run>_launch_full.log, report, do NOT
   relaunch unasked. If a run stopped early, judge the checkpoints that exist (trim the script's
   U list); it hard-exits on a missing scp.
2. Compare (control first): cd repo && uv run python $C/session_scripts_20261108/round_report.py
   tw_s3_u600 od1_u1200 gb1_u2000 nc3 nv69_u2000 nv70_u2000 ; append to decisions.md with
   hashes (append-only). Then draft the James reply for Julian's review — only after these land.
3. Report to Julian; he owes James a reply on the v29 PR conditions (continue lineage vs close).
   Ask Julian: reply text OK? Has James signed off the v70 lineage yet?
4. Still unbuilt: review item 2 (randomization+noise+delay training; fetching PR #94 for the
   table2_v1 preset touches the remote — ask first) and item 3 (native push/payload evals).
   Ask Julian before building either.
5. NEVER push, even as backup — standing rule until Julian asks. site/project.json is a month
   behind the campaign on purpose: it updates at publication, not before.

## Traps
- The surrogate evaluator refuses existing output dirs (mkdir exist_ok=False): move stale
  folders aside (rename *_failed_...), never delete results.
- Eval envs lack telemetry toe_xyz_world before the first control — odometry reads
  current["toe_body"] (commit cf9f7318). Don't "fix" that back.
- zsh ${U#00} strips one zero, not all: native_chain folders are nc2_u0600/nc3_u0600.
- Native init needs .pt + .json declaration whose checkpoint_sha256 names the bytes AND matches
  the pt's embedded infos (scripts/train.py:157-165); prepare() mounts the init's dir at
  /checkpoint. Narrowing recipe: $C/starts/make_nc3_ghost_critic_start.py.
- Viewer serves stale JSON on same-URL reload: cache-bust the /@fs URL (append %3Fv%3D2).
- graphify: trust graph only if graphify-out/graph.json newer than HEAD; run graphify update .
  after code changes. Grep with explicit paths, exclude .venv/, graphify-out/, tmp/.
- Spark mirror has no git metadata; keep the historical heartbeat paused; keep both GPU locks
  and the retained host reservation (docs/OPERATIONS.md).
- "Sim2real for legged robots.md" (repo root, untracked) is Julian's research report — leave it.
