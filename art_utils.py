"""Utility module for ART-based SAE feature taxonomy.

Provides a registry of ART modules, factory functions for easy instantiation,
and helpers for hierarchical clustering with SMART.
"""

import numpy as np
from artlib import FuzzyART, GaussianART, HypersphereART, BayesianART, EllipsoidART, SMART

try:
    from gpu_fuzzy_art import GPUFuzzyART, GPUSMART

    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False


# ---------------------------------------------------------------------------
# ART Module Registry
# ---------------------------------------------------------------------------

ART_REGISTRY = {
    "FuzzyART": {
        "class": FuzzyART,
        "default_params": {"rho": 0.7, "alpha": 0.01, "beta": 1.0},
        "complement_codes": True,
    },
    "GaussianART": {
        "class": GaussianART,
        "default_params": {"rho": 0.5, "alpha": 1e-10},
        "complement_codes": False,
        # sigma_init depends on data dimensionality
        "dim_dependent": lambda dim: {"sigma_init": np.ones(dim) * 0.5},
    },
    "HypersphereART": {
        "class": HypersphereART,
        "default_params": {"rho": 0.7, "alpha": 0.01, "beta": 1.0},
        "complement_codes": False,
        # r_hat must scale with dimensionality: max L2 distance in [0,1]^d = sqrt(d)
        "dim_dependent": lambda dim: {"r_hat": float(np.sqrt(dim))},
    },
    "BayesianART": {
        "class": BayesianART,
        "default_params": {"rho": 0.01},
        "complement_codes": False,
        # cov_init depends on data dimensionality
        "dim_dependent": lambda dim: {"cov_init": np.eye(dim) * 0.5},
    },
    "EllipsoidART": {
        "class": EllipsoidART,
        "default_params": {
            "rho": 0.7,
            "alpha": 1e-7,
            "beta": 1.0,
            "mu": 0.8,
        },
        "complement_codes": False,
        # r_hat must scale with dimensionality: max L2 distance in [0,1]^d = sqrt(d)
        "dim_dependent": lambda dim: {"r_hat": float(np.sqrt(dim))},
    },
}

if _GPU_AVAILABLE:
    ART_REGISTRY["GPUFuzzyART"] = {
        "class": GPUFuzzyART,
        "default_params": {"rho": 0.7, "alpha": 0.01, "beta": 1.0},
        "complement_codes": False,  # handled internally by prepare_data
    }


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def create_art_module(name, dim, overrides=None):
    """Instantiate an ART module by name with sensible defaults.

    Parameters
    ----------
    name : str
        Key in ART_REGISTRY (e.g. "HypersphereART").
    dim : int
        Data dimensionality (after PCA, before prepare_data).
    overrides : dict, optional
        Parameter overrides merged on top of defaults.

    Returns
    -------
    BaseART instance ready for prepare_data() and fit().
    """
    if name not in ART_REGISTRY:
        raise ValueError(
            f"Unknown module '{name}'. Available: {list(ART_REGISTRY.keys())}"
        )

    entry = ART_REGISTRY[name]
    params = dict(entry["default_params"])

    # Merge dimension-dependent params
    if "dim_dependent" in entry:
        params.update(entry["dim_dependent"](dim))

    # Merge user overrides last (highest priority)
    if overrides:
        params.update(overrides)

    return entry["class"](**params)


def create_smart_model(base_name, dim, rho_values, overrides=None):
    """Create a SMART hierarchical model using the named base ART class.

    Parameters
    ----------
    base_name : str
        Key in ART_REGISTRY for the base module.
    dim : int
        Data dimensionality (after PCA, before prepare_data).
    rho_values : list[float]
        Vigilance values per hierarchy level (monotonically increasing,
        except BayesianART which uses decreasing).
    overrides : dict, optional
        Parameter overrides for the base module (rho is excluded
        automatically since SMART sets it per level).

    Returns
    -------
    SMART instance ready for prepare_data() and fit().
    """
    if base_name not in ART_REGISTRY:
        raise ValueError(
            f"Unknown module '{base_name}'. Available: {list(ART_REGISTRY.keys())}"
        )

    entry = ART_REGISTRY[base_name]
    base_params = dict(entry["default_params"])

    # Merge dimension-dependent params
    if "dim_dependent" in entry:
        base_params.update(entry["dim_dependent"](dim))

    # Merge user overrides
    if overrides:
        base_params.update(overrides)

    # Remove rho — SMART sets it per level
    base_params.pop("rho", None)

    # GPUFuzzyART uses its own GPUSMART wrapper
    if _GPU_AVAILABLE and base_name == "GPUFuzzyART":
        return GPUSMART(rho_values=rho_values, **base_params)

    return SMART(entry["class"], rho_values, base_params)


def extract_hierarchy_labels(smart_model):
    """Extract per-level cluster labels from a fitted SMART model.

    Parameters
    ----------
    smart_model : SMART
        A fitted SMART instance.

    Returns
    -------
    np.ndarray of shape (n_samples, n_levels)
        Column i holds cluster labels at hierarchy level i.
    """
    n_levels = len(smart_model.modules)
    columns = []
    for i in range(n_levels):
        columns.append(smart_model.modules[i].labels_)
    return np.column_stack(columns)


def list_available_modules():
    """Print a summary of all registered ART modules."""
    print(f"{'Module':<18} {'Complement Codes':<18} {'Default Params'}")
    print("-" * 80)
    for name, entry in ART_REGISTRY.items():
        cc = "Yes" if entry["complement_codes"] else "No"
        params = {k: v for k, v in entry["default_params"].items()}
        dim_note = " (+dim-dependent)" if "dim_dependent" in entry else ""
        print(f"{name:<18} {cc:<18} {params}{dim_note}")
