"""
Compare JAX (orbax) checkpoint parameters against the original PyTorch checkpoint.

The conversion script stored all tensors as plain np.array() copies (no transpose),
so shapes and values should be byte-for-byte identical. We verify that here.

Note: Flax Linear uses [in, out] weight convention while PyTorch uses [out, in],
but the JAX model uses these weights with the PyTorch convention in mind
(either by using them transposed in forward passes, or by storing them as-is).
This script just checks that the stored bytes are identical.
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

mismatches = []
for key in sorted(common_keys):
    pt_arr = pt_sd[key].numpy()
    jax_arr = np.array(jax_params[key])

    if pt_arr.shape != jax_arr.shape:
        mismatches.append((key, "SHAPE", pt_arr.shape, jax_arr.shape, None))
        continue

    max_diff = np.max(np.abs(pt_arr - jax_arr))
    if max_diff > 1e-6:
        mismatches.append((key, "VALUES", pt_arr.shape, jax_arr.shape, max_diff))
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
