# ART-Organized SAE Feature Taxonomy

Cluster pre-trained SAE decoder weight vectors from **Pythia-70M** using Adaptive Resonance Theory (ART) to build hierarchical feature taxonomies.

## Prerequisites

- Python 3.10+
- GPU optional (ART clustering is CPU-based; SAE loading benefits from GPU)

## Quick Start

### Google Colab
1. Upload `art_utils.py` and open `sae_feature_taxonomy.ipynb`
2. Run all cells — dependencies install automatically
3. Default config uses `QUICK_TEST=True` (1000 features, fast)

### Local
```bash
pip install -r requirements.txt
jupyter notebook sae_feature_taxonomy.ipynb
```

## Module Comparison

| Module | Best For | Key Params | Complement Coding | Notes |
|--------|----------|------------|-------------------|-------|
| **HypersphereART** | General use (default) | `rho`, `alpha`, `beta`, `r_hat` | No | Spherical clusters, familiar geometry |
| FuzzyART | Low-dim data | `rho`, `alpha`, `beta` | **Yes** (doubles dims) | Classic ART; complement coding can be slow at high dims |
| GaussianART | Axis-aligned clusters | `rho`, `sigma_init` | No | Needs good `sigma_init` estimate |
| BayesianART | Full covariance | `rho`, `cov_init` | No | **Inverted vigilance** (lower rho = tighter clusters) |
| EllipsoidART | Elongated clusters | `rho`, `alpha`, `beta`, `mu`, `r_hat` | No | `mu` controls axis ratio |

## How to Swap Modules

Change **one string** in the Configuration cell:

```python
MODULE_NAME = "FuzzyART"  # was "HypersphereART"
```

The factory function handles all dimension-dependent parameters automatically.

## Vigilance Tuning Guide

Vigilance (`rho`) is ART's single knob for cluster granularity:

- **Higher rho → more clusters** (stricter match required)
- **Lower rho → fewer clusters** (looser matching)
- Start with the default, then use the vigilance sweep cell to explore

**BayesianART exception:** Vigilance is inverted — *lower* `rho` values produce *tighter* clusters. For SMART with BayesianART, use *decreasing* rho values (e.g., `[0.1, 0.01, 0.001]`).

## Expected Outputs

- **Flat clustering:** Cluster count depends on vigilance; expect 10–500 clusters for 1000 features
- **UMAP plot:** 2D scatter colored by cluster ID
- **SMART hierarchy:** 3-level taxonomy (coarse → medium → fine)
- **Sunburst chart:** Interactive plotly visualization of the hierarchy
- **Token interpretation:** Top tokens aligned with each cluster's mean direction

## PCA Recommendations

| Data Size | PCA Dims | Rationale |
|-----------|----------|-----------|
| Quick test (1K) | 50 | Fast iteration, captures most variance |
| Full (32K) | 50–100 | Balance speed and fidelity |
| None | Set `PCA_DIMS = None` | Full 512 dims; slower but no information loss |

## Troubleshooting

**OOM on Colab free tier:**
- Set `QUICK_TEST = True` (1000 features)
- Reduce `PCA_DIMS` to 30

**Too many clusters:**
- Lower `rho` (e.g., 0.3 instead of 0.7)
- Increase `r_hat` for HypersphereART/EllipsoidART

**Too few clusters:**
- Raise `rho` (e.g., 0.9)
- Decrease `r_hat`

**Slow clustering:**
- Use PCA to reduce dimensions
- Enable `QUICK_TEST` mode
- Avoid FuzzyART at high dims (complement coding doubles dimensions)
