import torch
import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path

# download original parameters with the .sh script that is already provided
pt_checkpoint_path = "model_params/ligandmpnn_v_32_030_25.pt"

parent_out_dir = Path("jax_parameters")
parent_out_dir.mkdir(exist_ok=True)

out_dir = parent_out_dir / "ligandmpnn_v_32_030_25_jax"

pt_checkpoint = torch.load(pt_checkpoint_path, map_location="cpu")
print(type(pt_checkpoint))
print(pt_checkpoint.keys())

state_dict = pt_checkpoint["model_state_dict"]

jax_params = {
    k: np.array(v.numpy()) for k, v in state_dict.items() if isinstance(v, torch.Tensor)
}

checkpoint_data = {
    "params": jax_params,
    "num_edges": pt_checkpoint["num_edges"],
    "noise_level": pt_checkpoint["noise_level"],
    "atom_context_num": pt_checkpoint["atom_context_num"],
}

checkpointer = ocp.StandardCheckpointer()
checkpointer.save(out_dir.resolve(), checkpoint_data)
checkpointer.wait_until_finished()
print(f"Saved orbax checkpoint to {out_dir}")
