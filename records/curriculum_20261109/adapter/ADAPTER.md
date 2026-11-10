# The sim2real adapter: what it is, what it proved, how to use it

Everything here is from the 2026-11-03 entries of `../decisions.md` (the hash-bound record);
code is on this branch in `src/locomotion/rl/sim2real.py` (commits b0f9f311, 37b26937,
84943803, b813c7f3, d32f5d84). Checkpoints in this folder are each arm's 200-update leg with
its declaration; `judges/` holds the three evaluation scripts; `readouts/` the identification
(R2) and matched-draw robustness summaries; `rehearsal_001/` a full deployment-loop record.

## Mechanism
`CompensatedActor` wraps a FROZEN core policy (u600_002, e29be0bc..., bytes sha-verified at
construction) with bounded, latent-conditioned gains/biases on the proprio + executed-target
input columns and a bounded pre-tanh action delta. Heads are zero-initialized, so the
composite equals the core bitwise at update 0 (unit-proven). PPO trains only the compensator;
`WarmStartedCritic` starts from the core's critic. `DynamicsRandomization` (surrogate env)
draws per-episode friction / mass / motor / kd scales, a 0-1 control action delay, and an
in-episode sag drain, from per-(seed,row,episode) streams; `randomization=None` is bitwise
inert (test_randomization). No gate, manifest or scorer was touched.

Training recipe (all arms): v35 task+reward, mixed_v3 terrain, seed 20260917, ranges friction
(0.6,1.4), mass (0.9,1.1), motor (0.9,1.1), kd (0.8,1.2), delay <= 1 control, sag (0,0.1)/min
floor 0.85. Flags: `--wrapper locomotion.rl.sim2real:Sim2RealVecEnv`; `--latent-supervision
1e-3` adds the identification loss (default 0 = stock learner); `identification=True` gives
the supervised instrument its own encoder off the control path (arm c); `condition_heads`
feeds the heads that latent detached (arm d). 200 updates is the drift-safe horizon: at u1000
the known surrogate drift tax dominates (arm a: full-96 24/96, omni 0/16).

## The four arms and the verdict
| arm | design | scored robust /48 (core 31) | hill-7 /36 (core 30) | omni15 /16 (core 4) | readout |
|---|---|---|---|---|---|
| a | reward-only | 32 | 30 | 3 | null (R2<=0.08) |
| b | coupled supervision | 32 | 23 (latent warped behavior) | 6 | mass 0.449 |
| c | decoupled instrument | 27 | 30 (restored) | 5 | mass 0.272, clean |
| d | conditioned heads (RMA-style) | 28 | 26 | 5 | mass 0.252 stable |

Conclusion (decisions.md, "PHASE 1 SURROGATE CONCLUSION"): within the trained ranges the core
is ALREADY robust; the bounded compensator cannot beat it (the scored half swings ±5/48 across
identical recipes — noise), and beyond the safe horizon it just channels the drift tax. The
real deliverable is the INSTRUMENT: mass_scale identifies at R2 ~0.25 from a 0.1 s proprio
window; friction/kd are marginal; motor, delay and sag are NOT identifiable at walking
excitation (torque ceiling rarely binds, one-control delay hides under velocity noise, 32 s
sag needs memory). s2r_c is the standing architecture: behavior preserved, instrument clean.

## How to use it (the deployment loop, rehearsed end to end)
`judges/sim2real_rehearsal.py` runs the whole loop against a hidden world W*; rehearsal_001/
is a complete record (W* friction 1.148, mass 0.940, motor 0.920, kd 1.024, delay 1, sag
0.085/min):
1. DEPLOY the composite (s2r_e, c-architecture) in W*: flat 6/6, zero falls (hill-7 stalls
   exactly where the bare core stalls — a weak-sagging-motor world binds both).
2. IDENTIFY: ~64 walking episodes through the instrument encoder; trust bar R2 >= 0.10 admits
   mass only. Measured: mass 0.992 vs truth 0.940 — beats the prior center; the rest sit at
   the prior (uninformative), exactly as phase 1 predicted.
3. HARDCODE (real2sim): write the TRUSTED identified parameters back into the simulator and
   re-tune there — do NOT ship the opaque compensator as a patch; the four-arm evidence says
   bounded output patching has no headroom over the robust core.
4. VALIDATE: compare calibrated-sim behavior to the real traces (rehearsal: flat gap 0.010 m).
Judges for any new arm, in order: `sim2real_readout.py` held-out R2 (did identification
emerge), omni15 nominal guard >= core's 4/16, then `sim2real_robust_eval.py` scored half on
matched draws vs the core's recorded 31/48 and its hill-7 walked half vs 30/36.

## On real hardware (phase 2, recorded plan)
- The adapter is an instrument first. Identifying motor strength, delay and sag needs
  EXCITATION or MEMORY, not more reward: a short calibration protocol (high-acceleration
  bench moves, an SPI-Active-style minute) and/or the stack's existing proprio_history
  observation group. Both are input-side changes: they go to James via Julian BEFORE any
  native run.
- Loop on the robot: walk -> identify (trusted axes only) -> write back into the sim ->
  fine-tune/validate in the calibrated sim -> redeploy. Combine with the v70 odometry input
  stack (see ../../HANDOFF.md "Connecting a trained checkpoint to a real hexapod") — the
  adapter calibrates the simulator; v70 makes the policy's own inputs robot-obtainable.
- Note: these arms wrap the u600_002-era core. To use the instrument with the current od1/v70
  lineage, retrain the compensator around the new frozen core (same recipe, 200 updates,
  c architecture); the machinery is lineage-agnostic.
