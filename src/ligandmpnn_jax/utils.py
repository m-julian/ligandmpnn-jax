from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from .proteinmpnn_model import ProteinMPNN
from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path
import jax.numpy as jnp


def make_fasta_seq(
    seq_arr: np.ndarray,
    protein_dict: dict,
    fasta_separation: str = ":",
):
    seq_out_str = []
    for mask in protein_dict["mask_c"]:
        seq_out_str += list(seq_arr[np.array(mask)])
        seq_out_str += [fasta_separation]
    return "".join(seq_out_str)[:-1]


def protein_dict_to_serializable(protein_dict: Mapping[str, Any]) -> dict:
    result = {}
    for key, value in protein_dict.items():
        if isinstance(value, jax.Array):
            result[key] = np.array(value).tolist()
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, list) and isinstance(value[0], jax.Array):
            result[key] = [np.array(i).tolist() for i in value]
        else:
            result[key] = value
    return result


def load_model(checkpoint_path: Path, rngs: nnx.Rngs, num_edges: int) -> ProteinMPNN:

    # TODO: should these be hardcoded, need to check other models as well.
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=num_edges,
        rngs=rngs,
    )

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path.resolve())
    params = restored["params"]

    # Walk the nnx Param leaves and set values from the flat PT-keyed checkpoint dict.
    # nnx uses 'kernel' for Linear weight, 'embedding' for Embed weight, 'scale' for
    # LayerNorm weight, 'bias' for all biases. PT keys use 'weight'/'bias'.
    def nnx_path_to_pt_key(path: tuple) -> str:
        parts = [str(p) for p in path]
        if parts[-1] in ("kernel", "embedding", "scale"):
            parts[-1] = "weight"
        return ".".join(parts)

    missing = []
    for path, var in nnx.state(model, nnx.Param).flat_state():
        pt_key = nnx_path_to_pt_key(path)
        if pt_key in params:
            var.value = jnp.array(params[pt_key])
        else:
            missing.append((path, pt_key))

    if missing:
        print(f"[WARN] {len(missing)} model params not found in checkpoint:")
        for path, pt_key in missing:
            print(f"  {'.'.join(str(p) for p in path)} -> '{pt_key}'")

    return model
