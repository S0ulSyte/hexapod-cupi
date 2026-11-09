"""DIAGNOSTIC: judge od1_u1200 with the odometry's foot-contact sense removed.

The eval tracker's observed() is patched to return contact=None, which keeps the stance set
from reset (all toes) forever — the no-sensor fallback, so every toe votes, swing legs
included. Outputs land in *_nocontact folders; this changes no registered input and judges
nothing for the goal.
"""

import runpy
import sys
from pathlib import Path

import locomotion.omniwalk.mdp.odometry as odometry

C = Path("/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum")
CKPT = str(C / "od1/checkpoint_update001200.pt")

real_observed = odometry.observed


def observed_nocontact(env):
    root, toes, _ = real_observed(env)
    return root, toes, None


odometry.observed = observed_nocontact

which = sys.argv[1]
if which == "full96":
    sys.argv = ["full96_suite", "--checkpoint", CKPT, "--output", str(C / "full96_od1_u1200_nocontact")]
    runpy.run_path(str(C / "full96_suite.py"), run_name="__main__")
else:
    sys.argv = ["omni15_suite", "--checkpoint", CKPT, "--output", str(C / "omni15_od1_u1200_nocontact")]
    runpy.run_path(str(C / "omni15_suite.py"), run_name="__main__")
