"""Make a ghost_v11 starting checkpoint from the v9 (ghost) checkpoint.

The ghost block moves from 5 inputs [gap fwd, gap left, sin, cos, elapsed] to 9: [side fwd, side left, sin,
cos, elapsed, relative velocity fwd, left, along fwd, along left]. Side and along take the gap's weights
(side + along = gap), the relative velocity starts at zero, so the first layer computes what v9's did.
Adam moments of the new columns start at zero. Usage: python make_ghost_v11_start.py <v9.pt> <out.pt>
"""

import sys

import torch

# First input of the ghost block: after 231 actor (234 critic) values and the 2 gait-clock values.
BLOCK = {"actor_state_dict": 233, "critic_state_dict": 236}


def widen(weight, start):
    gap = weight[:, start : start + 2]
    zeros = weight.new_zeros(weight.shape[0], 2)
    return torch.cat((weight, zeros, gap.clone()), 1)


def main(source, output):
    loaded = torch.load(source, weights_only=False, map_location="cpu")
    old = {key: loaded[key]["mlp.0.weight"].clone() for key in BLOCK}
    for key, start in BLOCK.items():
        weight = loaded[key]["mlp.0.weight"]
        if weight.shape[1] != start + 5:
            raise ValueError(f"{key} has {weight.shape[1]} inputs, not a ghost model's {start + 5}")
        loaded[key]["mlp.0.weight"] = widen(weight, start)
    for state in loaded["optimizer_state_dict"]["state"].values():
        for moment in ("exp_avg", "exp_avg_sq"):
            value = state.get(moment)
            if value is not None and value.ndim == 2 and value.shape[1] in (238, 241):
                state[moment] = torch.cat((value, value.new_zeros(value.shape[0], 4)), 1)
    # The new first layer on [x, relative velocity, along] with side = gap - along equals v9's on x.
    for key, start in BLOCK.items():
        x = torch.randn(64, start + 5)
        along = torch.randn(64, 2)
        new_x = x.clone()
        new_x[:, start : start + 2] -= along
        new_x = torch.cat((new_x, torch.randn(64, 2), along), 1)
        error = (new_x @ loaded[key]["mlp.0.weight"].T - x @ old[key].T).abs().max()
        print(f"{key}: {tuple(old[key].shape)} -> {tuple(loaded[key]['mlp.0.weight'].shape)}, max difference {float(error):.2e}")
        if error > 1e-4:
            raise ValueError("The widened layer does not reproduce v9")
    torch.save(loaded, output)


if __name__ == "__main__":
    main(*sys.argv[1:3])
