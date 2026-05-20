import torch
import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path

# download original parameters with the .sh script that is already provided
# to a directory. Then modify this line
pt_checkpoint_path = "pytorch_model_params/ligandmpnn_v_32_030_25.pt"

parent_out_dir = Path("jax_parameters")
parent_out_dir.mkdir(exist_ok=True)

out_dir = parent_out_dir / "ligandmpnn_v_32_030_25_jax"

pt_checkpoint = torch.load(pt_checkpoint_path, map_location="cpu")
print(type(pt_checkpoint))
print(pt_checkpoint.keys())

state_dict = pt_checkpoint["model_state_dict"]


def convert_tensor(k: str, v: torch.Tensor) -> np.ndarray:
    arr = v.numpy()
    # PyTorch Linear weight is [out, in]; Flax Linear kernel is [in, out]
    if arr.ndim == 2 and k.endswith(".weight") and "embedding" not in k:
        arr = arr.T
    return np.array(arr)


jax_params = {
    k: convert_tensor(k, v)
    for k, v in state_dict.items()
    if isinstance(v, torch.Tensor)
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
