from collections.abc import Mapping
from typing import Any

import jax
import numpy as np


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
