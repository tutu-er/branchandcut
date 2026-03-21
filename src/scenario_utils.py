import numpy as np


def _to_float_array(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}")
    return arr


def has_meaningful_renewable_data(sample: dict, tol: float = 1e-9) -> bool:
    renewable_data = _to_float_array(sample.get("renewable_data"))
    if renewable_data is None:
        return False
    return bool(np.any(np.abs(renewable_data) > tol))


def normalize_sample_arrays(sample: dict) -> dict:
    """Normalize legacy/new scenario fields in-place.

    Supported fields:
    - ``pd_data``: legacy load matrix
    - ``load_data``: explicit load matrix
    - ``renewable_data``: renewable available injection matrix
    """
    load_data = _to_float_array(sample.get("load_data"))
    renewable_data = _to_float_array(sample.get("renewable_data"))
    pd_data = _to_float_array(sample.get("pd_data"))

    if load_data is None and pd_data is not None:
        load_data = pd_data.copy()

    if pd_data is None and load_data is not None:
        pd_data = load_data.copy()

    if load_data is not None and renewable_data is not None and load_data.shape != renewable_data.shape:
        raise ValueError(
            f"load_data shape {load_data.shape} does not match renewable_data shape {renewable_data.shape}"
        )

    if load_data is not None and pd_data is not None and load_data.shape != pd_data.shape:
        raise ValueError(
            f"load_data shape {load_data.shape} does not match pd_data shape {pd_data.shape}"
        )

    if load_data is not None:
        sample["load_data"] = load_data
    if renewable_data is not None:
        sample["renewable_data"] = renewable_data
    if pd_data is not None:
        sample["pd_data"] = pd_data
    return sample


def get_sample_load_data(sample: dict) -> np.ndarray:
    normalize_sample_arrays(sample)
    if "load_data" in sample:
        return sample["load_data"]
    if "pd_data" in sample:
        return sample["pd_data"]
    raise KeyError("Sample does not contain load_data or pd_data")


def get_sample_renewable_data(sample: dict) -> np.ndarray:
    normalize_sample_arrays(sample)
    renewable_data = sample.get("renewable_data")
    if renewable_data is None:
        load_data = get_sample_load_data(sample)
        renewable_data = np.zeros_like(load_data)
    return renewable_data


def get_sample_net_load(sample: dict) -> np.ndarray:
    normalize_sample_arrays(sample)
    return get_sample_load_data(sample) - get_sample_renewable_data(sample)


def get_feature_vector_from_sample(sample: dict) -> np.ndarray:
    """Return NN features.

    New-format samples use ``[load, renewable]`` concatenation.
    Legacy samples fall back to flattened ``pd_data``.
    """
    has_explicit_load = "load_data" in sample
    has_explicit_renewable = "renewable_data" in sample
    normalize_sample_arrays(sample)
    if has_explicit_load and has_explicit_renewable:
        return np.concatenate(
            [
                sample["load_data"].flatten(),
                sample["renewable_data"].flatten(),
            ]
        )
    return get_sample_net_load(sample).flatten()


def get_feature_vector(pd_data: np.ndarray, renewable_data: np.ndarray | None = None) -> np.ndarray:
    """Return a flattened feature vector from explicit arrays."""
    pd_arr = _to_float_array(pd_data)
    if renewable_data is None:
        return pd_arr.flatten()
    renewable_arr = _to_float_array(renewable_data)
    if pd_arr.shape != renewable_arr.shape:
        raise ValueError(
            f"pd_data shape {pd_arr.shape} does not match renewable_data shape {renewable_arr.shape}"
        )
    return np.concatenate([pd_arr.flatten(), renewable_arr.flatten()])

