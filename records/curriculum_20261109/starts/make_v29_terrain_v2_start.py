"""Make a terrain_v2 starting checkpoint from Spark v29 u600 (ghost_v11).

terrain_v2 appends 36 inputs after the nine ghost_v11 values: terrain_v1's 30 footfall
columns plus a 6-point forward height profile along the command bearing (actor
242 -> 278, critic 245 -> 281). Every new first-layer column starts at zero, and the new
columns' Adam moments start at zero, so the widened network computes exactly what v29's
did until training moves it. Usage: python make_v29_terrain_v2_start.py <v29.pt> <out.pt>

Writes <out>.json beside the output with the derivation (source, widths, check).
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import torch

NEW = 36
OLD = {"actor_state_dict": 242, "critic_state_dict": 245}


def main(source, output):
    source, output = Path(source), Path(output)
    loaded = torch.load(source, weights_only=False, map_location="cpu")
    old = {key: loaded[key]["mlp.0.weight"].clone() for key in OLD}
    for key, width in OLD.items():
        weight = loaded[key]["mlp.0.weight"]
        if weight.shape[1] != width:
            raise ValueError(f"{key} has {weight.shape[1]} inputs, not a ghost_v11 model's {width}")
        loaded[key]["mlp.0.weight"] = torch.cat(
            (weight, weight.new_zeros(weight.shape[0], NEW)), 1
        )
    padded_moments = 0
    for state in loaded["optimizer_state_dict"]["state"].values():
        for moment in ("exp_avg", "exp_avg_sq"):
            value = state.get(moment)
            if value is not None and value.ndim == 2 and value.shape[1] in OLD.values():
                state[moment] = torch.cat((value, value.new_zeros(value.shape[0], NEW)), 1)
                padded_moments += 1
    checks = {}
    for key, width in OLD.items():
        x = torch.randn(64, width, dtype=torch.float64)
        wide = torch.cat((x, torch.randn(64, NEW, dtype=torch.float64) * 0), 1)
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
        "made_by": "runs/curriculum/starts/make_v29_terrain_v2_start.py",
        "source_checkpoint": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "widening": {
            "new_inputs": NEW,
            "actor_inputs": "242 -> 278",
            "critic_inputs": "245 -> 281",
            "new_columns": "zero weights, zero Adam moments",
            "optimizer_moments_padded": padded_moments,
        },
        "reproduction_check_max_abs_error": checks,
        "intended_wrapper_options": {
            "destination_input": "terrain_v2",
            "gait_clock": 60,
            "gait_clock_scaling": "speed_v1",
        },
        # The native loader reads the declaration as the checkpoint record: it requires
        # checkpoint_sha256 plus the identity and updates embedded in the .pt's infos.
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "identity": json.loads(json.dumps(loaded["infos"]["identity"])),
        "updates": loaded["infos"]["updates"],
    }
    output.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    print(f"wrote {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
