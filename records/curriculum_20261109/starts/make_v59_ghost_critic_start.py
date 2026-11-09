"""Make a terrain_v2_ghost_critic starting checkpoint from a terrain_v2 one (review item 1).

The actor's nine ghost_v11 input columns (233..241: after the 231 base values and the two
clock values, before the 36 terrain values) come from the simulator's world pose, which the
robot cannot supply, so they are dropped from the actor's first layer and its Adam moments
(actor 278 -> 269). The critic keeps its full 281-input privileged set untouched. The
narrowed actor computes exactly what the source computed on inputs whose ghost values are
zero, checked below to 1e-6. Usage:

  python make_v59_ghost_critic_start.py <terrain_v2.pt> <out.pt>

Writes <out>.json beside the output with the derivation (source, widths, check).
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import torch

GHOST = 9
START = 233  # 231 base + 2 clock values; the ghost block ends where the terrain block begins
OLD, NEW = 278, 269


def main(source, output):
    source, output = Path(source), Path(output)
    loaded = torch.load(source, weights_only=False, map_location="cpu")
    weight = loaded["actor_state_dict"]["mlp.0.weight"]
    if weight.shape[1] != OLD:
        raise ValueError(f"actor has {weight.shape[1]} inputs, not a terrain_v2 model's {OLD}")
    old = weight.clone()
    keep = torch.cat((torch.arange(START), torch.arange(START + GHOST, OLD)))
    loaded["actor_state_dict"]["mlp.0.weight"] = weight[:, keep].clone()
    sliced_moments = 0
    for state in loaded["optimizer_state_dict"]["state"].values():
        for moment in ("exp_avg", "exp_avg_sq"):
            value = state.get(moment)
            if value is not None and value.ndim == 2 and value.shape[1] == OLD:
                state[moment] = value[:, keep].clone()
                sliced_moments += 1
    x = torch.randn(64, OLD, dtype=torch.float64)
    x[:, START : START + GHOST] = 0.0
    error = (
        (
            x[:, keep] @ loaded["actor_state_dict"]["mlp.0.weight"].double().T
            - x @ old.double().T
        )
        .abs()
        .max()
    )
    print(f"actor: {tuple(old.shape)} -> {tuple(loaded['actor_state_dict']['mlp.0.weight'].shape)}, "
          f"max difference on ghost-zero inputs {float(error):.2e}")
    if error > 1e-6:
        raise ValueError("The narrowed layer does not reproduce the ghost-zero source to 1e-6")
    if loaded["critic_state_dict"]["mlp.0.weight"].shape[1] != 281:
        raise ValueError("critic is not a terrain_v2 model's 281 inputs")
    torch.save(loaded, output)
    note = {
        "date": str(date.today()),
        "made_by": "runs/curriculum/starts/make_v59_ghost_critic_start.py",
        "source_checkpoint": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "narrowing": {
            "dropped_inputs": GHOST,
            "dropped_columns": f"{START}..{START + GHOST - 1} (the nine ghost_v11 values)",
            "actor_inputs": f"{OLD} -> {NEW}",
            "critic_inputs": "281 unchanged (privileged ghost kept)",
            "optimizer_moments_sliced": sliced_moments,
        },
        "reproduction_check_max_abs_error": {"actor_state_dict_on_ghost_zero_inputs": float(error)},
        "intended_wrapper_options": {
            "destination_input": "terrain_v2_ghost_critic",
            "gait_clock": 60,
            "gait_clock_scaling": "speed_v1",
        },
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "identity": json.loads(json.dumps(loaded["infos"]["identity"])),
        "updates": loaded["infos"]["updates"],
    }
    output.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    print(f"wrote {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
