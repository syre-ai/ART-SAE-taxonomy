# ART-SAE Feature Taxonomy: Project Plan

## Current State

We cluster pre-trained SAE decoder weight vectors (W_dec) from Pythia-70M using GPUFuzzyART + GPUSMART, then interpret clusters via differential logit lens. At rho=0.2/0.4/0.6, we get 13 L0 clusters with genuine semantic themes (commerce, governance, morphology, physical movement, ideology, crime). The pipeline works end-to-end.

### Known Issues
- **L0 catch-all**: The largest L0 cluster absorbs ~34% of features
- **Uniform L2 branching**: At the finest level, clusters split into equal-sized children (geometric, not semantic)
- **Small clusters degrade logit lens**: Averaging <20 decoder vectors produces noisy mean directions
- **Pythia-70M is too small**: 512-dim hidden space fragments concepts; 6 layers limit hierarchy depth
- **No quality metrics**: We have no quantitative measure of cluster quality beyond visual inspection

---

## Tier 1: Leverage ART's Unique Properties

*Low effort, high novelty. These use ART-specific capabilities unavailable in k-means/HDBSCAN.*

### T1.1: Bounding Box Feature Importance
**What**: Extract per-cluster hyperbox bounds from FuzzyART weights. Compute per-dimension "tightness" (box width). Tight dimensions = the cluster's defining directions.

**Why**: ART weight vectors ARE hyperboxes with explicit min/max per dimension. This is structural interpretability — no other clustering method gives you axis-aligned boundaries you can read off directly.

**Tasks**:
- [ ] Add `get_bounding_boxes()` method to GPUFuzzyART: extract `lower = w[:dim]`, `upper = 1 - w[dim:]`, `width = upper - lower`
- [ ] Compute per-cluster specificity score: `mean(width)` per cluster. Low = specific concept, high = catch-all
- [ ] Print specificity scores alongside cluster tokens in Cell 11
- [ ] Visualize: scatter plot of cluster size vs. specificity (identifies real concepts vs. catch-alls)

**Files**: `gpu_fuzzy_art.py`, `sae_feature_taxonomy.ipynb` (Cell 11)

### T1.2: Targeted Logit Lens via Tight Dimensions
**What**: Instead of averaging all 512 dims for logit lens, zero out dimensions where the cluster's bounding box is wide (non-selective). Project only through the tight dimensions.

**Why**: The mean of 512 dims washes out signal. If a cluster's box is tight on only 50 dims, those 50 carry all the semantic information. This is a more precise question: "what tokens are promoted by the dimensions this cluster is *specific about*?"

**Tasks**:
- [ ] After extracting bounding boxes (T1.1), select top-K tightest dims per cluster
- [ ] Add `get_cluster_tokens_targeted()` that masks non-selective dims before logit lens projection
- [ ] Compare targeted vs. differential logit lens output side-by-side
- [ ] Determine optimal K (try 50, 100, 200) by comparing semantic coherence

**Files**: `sae_feature_taxonomy.ipynb` (Cell 11)
**Depends on**: T1.1

### T1.3: Match/Activation Confidence Scores
**What**: Expose match values (M) and activation values (T) from GPUFuzzyART's fit. These are computed but discarded.

**Why**: M tells you how well each feature fits its assigned cluster (rho to 1.0). T margin (gap between winner and runner-up) tells you how unambiguous the assignment is. These are free confidence scores with geometric interpretation.

**Tasks**:
- [ ] Add `predict_with_confidence()` to GPUFuzzyART returning `(labels, match_values, T_values)`
- [ ] Store per-sample M and T during `fit()` (or recompute in a `predict` pass)
- [ ] Compute per-cluster stats: mean M, min M, mean T-margin
- [ ] Identify borderline features (M close to rho) and outlier features (low T-margin)
- [ ] Add confidence column to Cell 11 output

**Files**: `gpu_fuzzy_art.py`, `sae_feature_taxonomy.ipynb` (Cell 11)

### T1.4: iCVI-Guided Clustering
**What**: Use artlib's incremental Calinski-Harabasz index as a match-reset function. This rejects cluster assignments that would worsen overall clustering quality, independent of vigilance.

**Why**: The iCVI provides an *objective* quality signal that compensates for suboptimal rho. It prevents bad merges (assigning a feature to a cluster that doesn't want it) and can produce more variable, data-driven branching instead of uniform geometric splits.

**Background**: artlib provides `iCVIFuzzyART` (extends FuzzyART with incremental CH) and `CVIART` (wraps any BaseART with batch sklearn CVIs). The iCVI paper (da Silva, Rayapati, Wunsch 2022) shows this accelerates convergence and improves cluster validity.

**Tasks**:
- [ ] Test `artlib.iCVIFuzzyART` on the current W_dec data (CPU, small subsample first)
- [ ] Compare cluster count, sizes, and branching variance vs. plain GPUFuzzyART
- [ ] If results are promising, implement iCH tracking in GPUFuzzyART's `_step_fit`:
  - Track per-cluster: count (n), centroid (v), compactness (CP), gradient (G)
  - Track global: mean (mu), WGSS, criterion_value
  - Add CH check in match-reset: accept assignment only if CH improves
- [ ] Alternatively, use `CVIART` wrapper with artlib's FuzzyART for comparison (batch mode, slower but supports CH/DB/Silhouette)
- [ ] Integrate into SMART hierarchy: does iCVI guidance produce more semantically coherent L1/L2 splits?

**Files**: `gpu_fuzzy_art.py`, `art_utils.py`, `sae_feature_taxonomy.ipynb` (Cell 10)
**References**: artlib `CVIART`, `iCVIFuzzyART`, `iCVI_CH` classes; da Silva et al. IEEE TNNLS 2022

### T1.5: Critical Feature Attention Patterns
**What**: The fuzzy AND `min(x, w_j)` computed during matching reveals exactly which dimensions drove the assignment. The residual `x - min(x, w_j)` shows what the cluster ignores.

**Why**: This is Grossberg's "attended critical feature pattern." For each feature assigned to a cluster, we can say precisely: "dimensions 42, 117, and 305 were the critical match drivers." This gives per-feature explanations, not just per-cluster.

**Tasks**:
- [ ] After fit, recompute `min(x, w_j)` for each sample and its assigned cluster
- [ ] Aggregate per-cluster: which dimensions are consistently critical (high overlap across samples)?
- [ ] Compare with bounding box tightness (T1.1) — these should agree
- [ ] Visualize: heatmap of critical dimensions across clusters

**Files**: `gpu_fuzzy_art.py`, `sae_feature_taxonomy.ipynb` (new analysis cell)
**Depends on**: T1.1

---

## Tier 2: Better Interpretation

*Medium effort, significant quality improvement.*

### T2.1: Token Voting (Enriched Logit Lens)
**What**: For each feature in a cluster, compute its individual top-5 logit lens tokens. Count which tokens appear most frequently across the cluster. Rank by frequency.

**Why**: Avoids the mean-collapse problem entirely. If 80% of features in a cluster have "Monday" in their top-5, that's a strong signal — even if the mean direction doesn't surface it. This is a natural TF-IDF analog.

**Tasks**:
- [ ] Reuse the pre-computed `top5_ids` tensor from the junk filter cell (already have per-feature top-5)
- [ ] Add `get_cluster_token_votes()` function: count token occurrences across cluster members
- [ ] Show both differential logit lens AND token votes for each cluster
- [ ] Compare: which method produces more coherent labels?

**Files**: `sae_feature_taxonomy.ipynb` (Cell 11)

### T2.2: Intra-Cluster Quality Metrics
**What**: Compute per-cluster cohesion (avg pairwise cosine similarity), separation (distance to nearest neighbor centroid), and global silhouette score.

**Why**: Currently we have no quantitative measure of cluster quality. These metrics let us rank clusters by coherence, identify problematic clusters, and compare across rho settings.

**Tasks**:
- [ ] Compute per-cluster avg pairwise cosine similarity
- [ ] Compute per-cluster centroid-to-nearest-centroid cosine distance
- [ ] Compute global silhouette score (sklearn, cosine metric)
- [ ] Add to Cell 11 output: print metrics alongside each cluster
- [ ] Plot size vs. cohesion scatter to identify real concepts vs. catch-alls
- [ ] Compare metrics across different rho_values to find optimal hierarchy

**Files**: `sae_feature_taxonomy.ipynb` (new metrics cell after Cell 10)

### T2.3: LLM-Generated Cluster Labels
**What**: Feed each cluster's top differential tokens to an LLM and ask for a 2-5 word semantic label.

**Why**: Makes the taxonomy immediately human-readable. "L0:3 (2421 features): Physical/Bodily Movement" is much more useful than a token list.

**Tasks**:
- [ ] Write `label_cluster(tokens, n_features)` function that prompts an LLM
- [ ] Support both API-based (Claude/GPT) and local (Pythia itself) labeling
- [ ] Generate labels for all L0 clusters
- [ ] Store labels in a dict for use in visualizations
- [ ] Optional: use labels as input to SimpleARTMAP (connects to ENHANCEMENTS.md E4)

**Files**: `sae_feature_taxonomy.ipynb` (new cell after Cell 11)

### T2.4: Interactive HTML Hierarchy Browser
**What**: Generate a self-contained HTML file with a collapsible tree. Each node shows cluster label, top tokens, size, specificity score.

**Why**: The current text output is hard to navigate for hierarchies with hundreds of clusters. An interactive tree lets you explore breadth-first or drill into specific branches.

**Tasks**:
- [ ] Build tree data structure from `hierarchy` array + cluster metadata
- [ ] Generate collapsible HTML/JS tree (no server needed)
- [ ] Include per-node: LLM label (if available), token list, size, specificity, cohesion
- [ ] Save as `taxonomy_browser.html` in project root
- [ ] Optional: integrate with plotly sunburst for an alternative view

**Files**: `sae_feature_taxonomy.ipynb` (new cell), `taxonomy_browser.html` (generated)
**Depends on**: T2.2, T2.3

---

## Tier 3: Model & Data Upgrade

*Medium effort, foundational improvement for scientific rigor.*

### T3.1: Upgrade to Pythia-410M
**What**: Switch from Pythia-70M (512-dim, 32K SAE) to Pythia-410M (1024-dim, 65K SAE).

**Why**: Pythia-70M's 512-dim space fragments concepts. Studies show its SAE features map many-to-one onto larger model features. The 1024-dim space should produce cleaner, more coherent feature directions. Uses same sparsify loader — minimal code changes.

**Tasks**:
- [ ] Update config: `LAYER_IDX`, model name for `Sae.load_many("EleutherAI/sae-pythia-410m-65k")`
- [ ] Update LM loading: `AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")`
- [ ] Verify unembed architecture (tied vs untied embeddings)
- [ ] Adjust junk filter if vocabulary differs
- [ ] Re-run pipeline and compare taxonomy quality vs. Pythia-70M
- [ ] Document which layer(s) produce the most interpretable taxonomy

**Files**: `sae_feature_taxonomy.ipynb` (Cells 2, 3, 4)

### T3.2: Support Gemma Scope SAEs (via SAELens)
**What**: Add support for loading Google's Gemma Scope SAEs (Gemma-2-2B or Gemma-3-1B) alongside the current sparsify-based Pythia SAEs.

**Why**: Gemma Scope is the gold standard SAE ecosystem with SAEs at 16K/64K/256K/1M widths, every layer, residual+MLP+attention. The 2304-dim hidden space should produce dramatically richer feature hierarchies. Multiple dictionary sizes enable studying feature splitting (small SAE parent → large SAE children).

**Tasks**:
- [ ] Install SAELens: `pip install sae-lens`
- [ ] Add Gemma SAE loader alongside sparsify loader in Cell 3
- [ ] Handle architecture differences: Gemma uses tied embeddings? Different LN placement?
- [ ] Test with smallest Gemma Scope SAE (16K width) first
- [ ] Compare taxonomy quality across models (Pythia-70M vs 410M vs Gemma-2-2B)

**Files**: `sae_feature_taxonomy.ipynb` (Cells 1, 3, 4), possibly `art_utils.py`

### T3.3: Co-Activation Validation
**What**: Run a small corpus through the LM+SAE, record which features activate, compute co-activation matrix. Measure Adjusted Mutual Information (AMI) between ART geometric clusters and co-activation clusters.

**Why**: Li et al. (2025, "Geometry of Concepts") found that geometric clusters (cosine similarity on W_dec) align with functional clusters (co-activation) at 954 standard deviations above chance. Reproducing this validates that our ART clusters correspond to real functional groupings, not just geometric artifacts.

**Tasks**:
- [ ] Run ~1000 sequences from The Pile through Pythia-70M + SAE
- [ ] Record per-feature activation binary vectors (active/not per sequence)
- [ ] Compute co-activation affinity matrix (Jaccard or phi coefficient)
- [ ] Cluster co-activation matrix independently (HDBSCAN or spectral clustering)
- [ ] Compute AMI between ART clusters and co-activation clusters
- [ ] Report: "ART geometric clusters explain X% of functional co-activation structure"

**Files**: `sae_feature_taxonomy.ipynb` (new validation cell)
**References**: Li et al. 2025 "The Geometry of Concepts"

---

## Tier 4: ART-Specific Advanced Extensions

*These are the ENHANCEMENTS.md items, ordered by value for this project. See ENHANCEMENTS.md for full descriptions.*

### T4.1: Supervised Taxonomy via SimpleARTMAP (E4)
Use auto-generated semantic labels (from T2.3) as supervision. Match tracking forces geometric clusters to be semantically consistent. **Highest-priority enhancement** — tests whether ART's match tracking genuinely improves semantic coherence.

### T4.2: Co-Activation Topology via TopoART (E1)
Cluster features by co-activation patterns (from T3.3). TopoART produces an adjacency graph between clusters — which feature groups frequently co-activate. Complements the SMART hierarchy with behavioral relationships.

### T4.3: DualVigilanceART for Non-Convex Clusters (E7)
Wrap FuzzyART in DualVigilanceART. Upper vigilance controls fine-grained hyperboxes; lower vigilance merges them into abstract categories of arbitrary shape. May fix the catch-all problem (L0:0) by allowing elongated/non-convex cluster shapes.

### T4.4: Feature-Token Biclustering via BARTMAP (E3)
Simultaneously cluster features AND tokens using activation data. Produces (feature family, token family) pairs — Grossberg's bidirectional association.

### T4.5: Temporal Stability via Incremental Fit (E5)
Track feature emergence/death across Pythia training checkpoints using `partial_fit()`. Tests ART's stability-plasticity balance on real non-stationary data.

### T4.6: Cross-Layer Flow via ARTMAP (E2)
Build ART taxonomies at multiple layers. Use ARTMAP to learn cross-layer feature mappings. Reveals how feature families transform through the model.

### T4.7: Multi-Signal Fusion via FusionART (E6)
Fuse decoder geometry + activation behavior + token semantics into a single multi-channel taxonomy.

---

## Implementation Priority

**Phase A (Foundation)**: T1.1 → T1.2 → T2.1 → T2.2 → T1.3
*Goal*: Solid per-cluster quality metrics and multiple interpretation methods. ~1-2 days.

**Phase B (Quality)**: T1.4 → T2.3 → T2.4 → T1.5
*Goal*: iCVI-guided clustering, human-readable labels, interactive browser. ~2-3 days.

**Phase C (Scale)**: T3.1 → T3.2 → T3.3
*Goal*: Bigger models, validated results. ~2-3 days.

**Phase D (Novel ART)**: T4.1 → T4.3 → T4.2 → T4.4-T4.7
*Goal*: ART-specific innovations from ENHANCEMENTS.md. Ongoing.

---

## Key References

### SAE Interpretability
- Li et al. (2025). "The Geometry of Concepts: SAE Feature Structure." *Entropy*. [arXiv:2410.19750](https://arxiv.org/abs/2410.19750)
- Chanin et al. (2024). "A is for Absorption." NeurIPS 2024. [arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- Luo et al. (2026). "From Atoms to Trees: HSAE." [arXiv:2602.11881](https://arxiv.org/abs/2602.11881)
- Karvonen et al. (2025). "SAEBench." [arXiv:2503.09532](https://arxiv.org/abs/2503.09532)
- Korznikov et al. (2026). "Sanity Checks for SAEs." [arXiv:2602.14111](https://arxiv.org/abs/2602.14111)
- EleutherAI (2025). "Open Source Automated Interpretability." [Blog](https://blog.eleuther.ai/autointerp/)

### ART Theory
- Grossberg (2020). "A Path Toward Explainable AI." *Frontiers in Neurorobotics*. [DOI:10.3389/fnbot.2020.00036](https://www.frontiersin.org/articles/10.3389/fnbot.2020.00036)
- da Silva, Rayapati, Wunsch (2022). "iCVI-ARTMAP." *IEEE TNNLS* 34(12). [PubMed](https://pubmed.ncbi.nlm.nih.gov/35353707/)
- da Silva et al. (2019). "A Survey of ART Neural Network Models." [arXiv:1905.11437](https://arxiv.org/abs/1905.11437)
- Melton et al. (2025). "Deep ARTMAP." [arXiv:2503.07641](https://arxiv.org/abs/2503.07641)
- Carpenter & Tan (1995). "Rule Extraction from Neural Architecture." *Connection Science*.

### Available SAEs
- `EleutherAI/sae-pythia-70m-32k` — current (512-dim, 32K dict)
- `EleutherAI/sae-pythia-410m-65k` — recommended upgrade (1024-dim, 65K dict)
- `google/gemma-scope-2b-pt-res` — gold standard (2304-dim, 16K-65K dict)
- `google/gemma-scope-2-1b-pt` — newest (2304-dim, 16K-1M dict, crosscoders)
- `EleutherAI/sae-Llama-3.2-1B-131k` — alternative (2048-dim, 131K dict, gated access)
