"""Make a terrain_v2_critic starting checkpoint from Spark v29 u600 (ghost_v11).

terrain_v2_critic gives the 36 terrain values (terrain_v1's 30 footfall columns plus the 6-point
forward profile) to the CRITIC alone: the critic's first layer widens 245 -> 281 with zero columns
and zero Adam moments, and the actor stays exactly v29's 242 inputs - the ablation arm for the
native proposal. Usage: python make_v29_terrain_v2_critic_start.py <v29.pt> <out.pt>

Writes <out>.json beside the output with the derivation (source, widths, check).
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import torch

ADD = {"actor_state_dict": (242, 0), "critic_state_dict": (245, 36)}


def main(source, output):
    source, output = Path(source), Path(output)
    loaded = torch.load(source, weights_only=False, map_location="cpu")
    old = {key: loaded[key]["mlp.0.weight"].clone() for key in ADD}
    for key, (width, extra) in ADD.items():
        weight = loaded[key]["mlp.0.weight"]
        if weight.shape[1] != width:
            raise ValueError(f"{key} has {weight.shape[1]} inputs, not a ghost_v11 model's {width}")
        if extra:
            loaded[key]["mlp.0.weight"] = torch.cat(
                (weight, weight.new_zeros(weight.shape[0], extra)), 1
            )
    padded_moments = 0
    for state in loaded["optimizer_state_dict"]["state"].values():
        for moment in ("exp_avg", "exp_avg_sq"):
            value = state.get(moment)
            if value is not None and value.ndim == 2 and value.shape[1] == 245:
                state[moment] = torch.cat((value, value.new_zeros(value.shape[0], 36)), 1)
                padded_moments += 1
    checks = {}
    for key, (width, extra) in ADD.items():
        x = torch.randn(64, width, dtype=torch.float64)
        wide = torch.cat((x, torch.zeros(64, extra, dtype=torch.float64)), 1)
        error = (
            (wide @ loaded[key]["mlp.0.weight"].double().T - x @ old[key].double().T).abs().max()
        )
        checks[key] = float(error)
        print(
            f"{key}: {tuple(old[key].shape)} -> {tuple(loaded[key]['mlp.0.weight'].shape)},"
            f" max difference {float(error):.2e}"
        )
        if error > 1e-6:
            raise ValueError("The widened layer does not reproduce v29 to 1e-6")
    torch.save(loaded, output)
    note = {
        "date": str(date.today()),
        "made_by": "runs/curriculum/starts/make_v29_terrain_v2_critic_start.py",
        "source_checkpoint": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "widening": {
            "new_inputs": "critic only, 36",
            "actor_inputs": "242 (unchanged)",
            "critic_inputs": "245 -> 281",
            "new_columns": "zero weights, zero Adam moments",
            "optimizer_moments_padded": padded_moments,
        },
        "reproduction_check_max_abs_error": checks,
        "intended_wrapper_options": {
            "destination_input": "terrain_v2_critic",
            "gait_clock": 60,
            "gait_clock_scaling": "speed_v1",
        },
    }
    output.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    print(f"wrote {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
