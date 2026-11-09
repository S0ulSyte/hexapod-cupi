"""DIAGNOSTIC: judge od1_u1200 with foot contact INFERRED from the gait schedule.

No contact sensor: a toe counts as stance unless its tripod's swing window (phase fraction
< 0.4 for tripod A, 0.5..0.9 for tripod B, from the tracker's own speed-scaled gait phase)
says it is swinging while a command is active. Outputs land in *_gaitcontact folders; no
registered input changes.
"""

import runpy
import sys
from pathlib import Path

import torch

import locomotion.omniwalk.mdp.odometry as odom
from locomotion.omniwalk.mdp.destination import GHOST_V11_FEATURES
from locomotion.omniwalk.mdp.rewards import TRIPOD_A

C = Path("/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum")
CKPT = str(C / "od1/checkpoint_update001200.pt")
TA = torch.tensor(TRIPOD_A)
SWING = 0.4

real_observed = odom.observed


def inputs_gait_contact(self, pose, command, episode_steps):
    previous = self.inner.command.clone()
    full = self.inner.inputs(pose, command, episode_steps)
    root, toes, _ = real_observed(self.env)
    fraction = torch.remainder(self.inner.gait.phase.to(torch.float32), 1.0)
    swing_a = fraction < SWING
    swing_b = (fraction >= 0.5) & (fraction < 0.5 + SWING)
    swing = torch.where(TA[None, :], swing_a[:, None], swing_b[:, None])
    moving = (command != 0).any(-1, keepdim=True)
    contact = ~(swing & moving.to(swing.device))
    starts = episode_steps.to(self.odom.theta.device) == 0
    if self._fresh:
        self.odom.reset(torch.arange(len(pose)), root, toes, contact)
        self._fresh = False
    elif bool(starts.any()):
        self.odom.reset(starts.nonzero(as_tuple=False).flatten(), root, toes, contact)
    fresh = starts | (command.to(previous) != previous).any(-1)
    self.odom.advance(root, toes, contact, command, fresh)
    features = self.odom.features(fresh, self.inner.elapsed, command)
    return torch.cat((features.to(full), full[..., GHOST_V11_FEATURES:]), -1)


odom.OdomTerrainTracker.inputs = inputs_gait_contact

which = sys.argv[1]
if which == "full96":
    sys.argv = ["full96_suite", "--checkpoint", CKPT, "--output", str(C / "full96_od1_u1200_gaitcontact")]
    runpy.run_path(str(C / "full96_suite.py"), run_name="__main__")
else:
    sys.argv = ["omni15_suite", "--checkpoint", CKPT, "--output", str(C / "omni15_od1_u1200_gaitcontact")]
    runpy.run_path(str(C / "omni15_suite.py"), run_name="__main__")
