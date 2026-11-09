"""Make a terrain_v2_ghost_critic starting checkpoint from the native chain3 model, in the
native trainer's init format (review item 1, native replay).

Same narrowing as make_v59_ghost_critic_start.py — the actor's nine ghost_v11 input columns
(233..241) leave the first layer and its Adam moments (actor 278 -> 269), the critic keeps its
full 281-input privileged set — but the embedded ``infos`` declaration is kept byte-identical
and the output declaration JSON is the source's with only ``checkpoint_sha256`` rebound, so
scripts/train.py's require_starting_identity / require_embedded_declaration checks pass.

  python make_nc3_ghost_critic_start.py <chain3 checkpoint.pt> <out.pt>
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import torch

GHOST = 9
START = 233
OLD, NEW = 278, 269


def main(source, output):
    source, output = Path(source), Path(output)
    declaration = json.loads(source.with_suffix(".json").read_text())
    loaded = torch.load(source, weights_only=False, map_location="cpu")
    infos = loaded.get("infos")
    if infos != {"identity": declaration["identity"], "updates": declaration["updates"]}:
        raise ValueError("Source embedded declaration differs from its JSON record")
    weight = loaded["actor_state_dict"]["mlp.0.weight"]
    if weight.shape[1] != OLD:
        raise ValueError(f"actor has {weight.shape[1]} inputs, not a terrain_v2 model's {OLD}")
    if loaded["critic_state_dict"]["mlp.0.weight"].shape[1] != 281:
        raise ValueError("critic is not a terrain_v2 model's 281 inputs")
    old = weight.clone()
    keep = torch.cat((torch.arange(START), torch.arange(START + GHOST, OLD)))
    loaded["actor_state_dict"]["mlp.0.weight"] = weight[:, keep].clone()
    sliced = 0
    for state in loaded["optimizer_state_dict"]["state"].values():
        for moment in ("exp_avg", "exp_avg_sq"):
            value = state.get(moment)
            if value is not None and value.ndim == 2 and value.shape[1] == OLD:
                state[moment] = value[:, keep].clone()
                sliced += 1
    x = torch.randn(64, OLD, dtype=torch.float64)
    x[:, START : START + GHOST] = 0.0
    error = (
        (x[:, keep] @ loaded["actor_state_dict"]["mlp.0.weight"].double().T - x @ old.double().T)
        .abs()
        .max()
    )
    print(
        f"actor: {tuple(old.shape)} -> {tuple(loaded['actor_state_dict']['mlp.0.weight'].shape)}, "
        f"{sliced} moment tensors sliced, max difference on ghost-zero inputs {float(error):.2e}"
    )
    if error > 1e-6:
        raise ValueError("The narrowed layer does not reproduce the ghost-zero source to 1e-6")
    torch.save(loaded, output)
    declaration["checkpoint_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    declaration["narrowed_from"] = {
        "date": str(date.today()),
        "made_by": "runs/curriculum/starts/make_nc3_ghost_critic_start.py",
        "source_checkpoint": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dropped_columns": f"{START}..{START + GHOST - 1} (the nine ghost_v11 values)",
        "actor_inputs": f"{OLD} -> {NEW}",
    }
    output.with_suffix(".json").write_text(json.dumps(declaration, indent=1))
    print("checkpoint_sha256", declaration["checkpoint_sha256"])
    print("declaration_sha256", hashlib.sha256(output.with_suffix(".json").read_bytes()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
