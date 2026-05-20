"""
Compare JAX (orbax) checkpoint parameters against the original PyTorch checkpoint.

Linear weights are transposed during conversion (PyTorch [out, in] -> Flax [in, out]),
so we apply the same transform before comparing. All other tensors are stored as-is.
"""

import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path

try:
    import torch
    import jax
except ImportError:
    print("torch not found, cannot convert")
    quit()

# change model here
# orbax uses a directory to store model params
PT_PATH = Path("pytorch_model_params/ligandmpnn_v_32_030_25.pt")
JAX_PATH = Path("jax_parameters/ligandmpnn_v_32_030_25_jax")

pt_checkpoint = torch.load(PT_PATH, map_location="cpu")
pt_sd = pt_checkpoint["model_state_dict"]

checkpointer = ocp.StandardCheckpointer()
jax_ckpt = checkpointer.restore(JAX_PATH.resolve())
jax_params = jax_ckpt["params"]

pt_keys = set(pt_sd.keys())
jax_keys = set(jax_params.keys())

missing_in_jax = pt_keys - jax_keys
extra_in_jax = jax_keys - pt_keys

if extra_in_jax or missing_in_jax:
    print("There are extra/missing params")
    print(f"extra params {extra_in_jax}")
    print(f"missing in jax {missing_in_jax}")

common_keys = pt_keys & jax_keys
print(f"\nComparing {len(common_keys)} shared parameters ...\n")


def expected_jax_arr(key: str, pt_arr: np.ndarray) -> np.ndarray:
    if pt_arr.ndim == 2 and key.endswith(".weight") and "embedding" not in key:
        return pt_arr.T
    return pt_arr


mismatches = []
for key in sorted(common_keys):
    pt_arr = pt_sd[key].numpy()
    jax_arr = np.array(jax_params[key])
    expected = expected_jax_arr(key, pt_arr)

    if expected.shape != jax_arr.shape:
        mismatches.append((key, "SHAPE", expected.shape, jax_arr.shape, None))
        continue

    max_diff = np.max(np.abs(expected - jax_arr))
    if max_diff > 1e-6:
        mismatches.append((key, "VALUES", expected.shape, jax_arr.shape, max_diff))
    else:
        print(f"  OK  {key:60s}  shape={jax_arr.shape}  max_diff={max_diff:.2e}")

print()
if mismatches:
    print(f"[FAIL] {len(mismatches)} parameter(s) did not match:\n")
    for key, reason, pt_shape, jax_shape, diff in mismatches:
        if reason == "SHAPE":
            print(f"  SHAPE MISMATCH  {key}")
            print(f"    PT: {pt_shape}   JAX: {jax_shape}")
        else:
            print(f"  VALUE MISMATCH  {key}  max_diff={diff:.2e}  shape={jax_shape}")
else:
    print(f"[PASS] All {len(common_keys)} shared parameters match (byte-identical).")
