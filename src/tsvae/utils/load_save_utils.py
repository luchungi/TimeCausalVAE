import pickle
import torch
import json
import numpy as np


def save_obj(obj: object, filepath: str):
    """Generic function to save an object with different methods."""
    if filepath.endswith("pkl"):
        saver = pickle.dump
    elif filepath.endswith("pt"):
        saver = torch.save
    elif filepath.endswith("json"):
        with open(filepath, "w") as f:
            json.dump(obj, f)
        return 0
    elif filepath.endswith("npy"):
        np.save(filepath, obj)
        return 0
    else:
        raise NotImplementedError(f"No suitable saver for the path: {filepath}")

    with open(filepath, "wb") as f:
        saver(obj, f)
    return 0


def load_obj(filepath):
    """Generic function to load an object."""
    if filepath.endswith("pkl"):
        loader = pickle.load
    elif filepath.endswith("pt"):
        loader = torch.load
        with open(filepath, "rb") as f:
            return torch.load(f,weights_only=True)
    elif filepath.endswith("json"):
        import json
        loader = json.load
    elif filepath.endswith("npy"):
        return np.load(filepath)
    else:
        raise NotImplementedError(f"No suitable loader for the path: {filepath}")
    with open(filepath, "rb") as f:
        return loader(f)
