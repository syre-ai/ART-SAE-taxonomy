# ART-SAE Taxonomy: Enhancement Proposal

*Inspired by Grossberg (2025) "ChatSOME" paper, grounded in recent SAE interpretability literature, using artlib's full capabilities.*

**Guiding principle**: ART *on top of* deep learning — using ART's strengths (principled granularity, stability-plasticity, online learning, explainability) to organize and interpret SAE features, not to replace transformers.

---

## Context: What the Literature Tells Us

### SAE Feature Organization is an Open Problem

- **Feature splitting & absorption** (Chanin et al., NeurIPS 2024): As SAE width increases, broad features split into sub-features, but parent features get "absorbed" — failing to fire where they should. Varying SAE size/sparsity doesn't fix this.
- **Feature instability** (EleutherAI, Dec 2024): SAEs trained on the same data with different seeds share only ~53% of features. The "true" feature basis is underdetermined.
- **Matryoshka SAEs** (ICML 2025): Nested dictionaries at increasing widths produce natural hierarchies, paralleling SMART's multi-vigilance approach.
- **HSAE** (Feb 2026): Hierarchical SAEs with explicit parent-child tree structure via structural constraint loss. Closest prior art to our goals, but purely neural.
- **Sparse Feature Coactivation** (arXiv 2506.18141, 2025): Co-activation of SAE features from just a handful of prompts reveals causal semantic modules. Concept features emerge early; relation features concentrate in later layers.
- **Sparse Feature Circuits** (Marks et al., ICLR 2025): Causally implicated subnetworks of SAE features form interpretable circuit graphs.
- **Crosscoders** (Anthropic, Oct 2024): Cross-layer SAE variants reveal features that persist across layers ("cross-layer superposition").

### ART Tools Available but Unused

artlib v0.1.7 provides modules we're not yet leveraging:

| Module | Purpose | Currently Used? |
|--------|---------|----------------|
| FuzzyART | Unsupervised hyper-box clustering | Yes (via GPUFuzzyART) |
| SMART | Hierarchical multi-vigilance | Yes (via GPUSMART) |
| **SimpleARTMAP** | Supervised clustering with label constraints | No |
| **ARTMAP** | Two-module supervised association learning | No |
| **DeepARTMAP** | Generalized hierarchical supervised/unsupervised | No |
| **TopoART** | Clustering + adjacency graph construction | No |
| **DualVigilanceART** | Non-convex cluster shapes via dual vigilance | No |
| **FusionART** | Multi-channel data fusion | No |
| **BARTMAP** | Biclustering (rows AND columns simultaneously) | No |
| **CVIART** | Cluster validity-guided ART | No |

### No Prior Work Combines ART with SAEs

Our search found zero papers applying ART to SAE feature organization. This is a genuinely novel direction. The closest bridges are:
- **Deep ARTMAP** (Melton et al., Mar 2025): Generalized hierarchical ART, implemented in artlib
- **DeepART** (Petrenko et al., Oct 2025): Treats deep network layers as FuzzyART modules
- **Grossberg (2020)**: Positions ART as a path toward explainable AI

---

## Proposed Enhancements

### Enhancement 1: Feature Co-activation Topology via TopoART

**Grossberg parallel**: ARTSCENE Search — objects and spatial contexts build an incrementally richer scene representation through learned adjacency.

**Motivation**: Sparse Feature Coactivation (2025) shows that co-firing patterns reveal semantic modules. Currently we cluster features by decoder weight similarity alone, ignoring *behavioral* relationships.

**Approach**:
1. Run a corpus of prompts through Pythia-70M and collect SAE activations
2. Build a feature co-activation matrix (which features fire together)
3. Use **TopoART** to cluster features based on co-activation patterns
4. TopoART naturally produces an **adjacency graph** between clusters (it records both first and second resonant categories), revealing which feature groups are neighbors in activation space

**What this adds**: A graph structure on top of the flat/hierarchical taxonomy. The SMART hierarchy tells us "feature A is a sub-type of feature B by geometric similarity." The TopoART graph tells us "feature group X frequently co-activates with feature group Y in practice." Together, these capture both the *what* (decoder geometry) and the *when* (activation behavior).

**artlib class**: `TopoART(base_module, beta_lower, tau, phi)`

---

### Enhancement 2: Cross-Layer Feature Flow via ARTMAP

**Grossberg parallel**: Hierarchical chunking across cortical levels — items at one level become the inputs to the next level's chunking process. Also: What/Where cortical streams processing different aspects of the same scene.

**Motivation**: Crosscoders (Anthropic, 2024) show that features persist, split, and merge across layers. Currently we only analyze one layer at a time.

**Approach**:
1. Extract SAE decoder weights at multiple layers (e.g., layers 1, 3, 5 of Pythia-70M)
2. For each adjacent pair (L, L+1):
   - **A-side**: FuzzyART clusters the layer-L decoder directions
   - **B-side**: FuzzyART clusters the layer-L+1 decoder directions
   - **Map field**: Learns which L-clusters predict which L+1-clusters
3. The map field `map_a2b()` then provides a **feature flow graph**: how feature families at one layer transform into feature families at the next

**What this adds**: A cross-layer story. Instead of separate per-layer taxonomies, we get a unified view of how feature types evolve through the model. Persistent features (same A→B mapping across many samples) represent stable computational primitives. Splitting features (one A-cluster → multiple B-clusters) indicate where the model refines representations.

**artlib class**: `ARTMAP(module_a, module_b)` or `DeepARTMAP` for the full stack

**Note**: The supervision signal comes from the data itself — features at layer L+1 are the "labels" for features at layer L. This is self-supervised in the spirit of Grossberg's bidirectional associative learning.

---

### Enhancement 3: Feature-Token Biclustering via BARTMAP

**Grossberg parallel**: Many-to-one and one-to-many associative maps — multiple visual categories mapping to one name, one image mapping to many descriptors.

**Motivation**: Our current interpretation (cell 11) uses a simple cosine similarity between cluster mean directions and token embeddings. This is one-directional and doesn't capture the structure of which token groups map to which feature groups.

**Approach**:
1. Construct a feature-activation matrix: rows = SAE features, columns = tokens from a corpus, values = activation strength
2. Feed this to **BARTMAP**:
   - `module_a` (FuzzyART) clusters the rows (features) by their activation profiles across tokens
   - `module_b` (FuzzyART) clusters the columns (tokens) by which features they activate
   - The `eta` parameter enforces minimum correlation within each bicluster
3. The result: simultaneous feature-groups and token-groups, with each bicluster being a (feature family, token family) pair

**What this adds**: Structured interpretation. Instead of "cluster 7's top tokens are [the, a, an]", we get "feature family F clusters with token family T, where F = {features 42, 107, 203} and T = {the, a, an, this, that}." This is Grossberg's bidirectional association: features predict tokens AND tokens predict features.

**artlib class**: `BARTMAP(module_a, module_b, eta)`

---

### Enhancement 4: Supervised Taxonomy via SimpleARTMAP

**Grossberg parallel**: Learning from a teacher — a teacher (auto-labeler) guides the learner's (ART's) category formation by providing linguistic descriptions of what is being perceived.

**Motivation**: Our current taxonomy is purely unsupervised. But we have access to automated feature labels (via top-activating tokens, or LLM-based auto-interpretation like EleutherAI's Delphi). These labels are noisy but informative.

**Approach**:
1. Generate semantic labels for each SAE feature:
   - Simple: top-5 activating tokens concatenated as a string
   - Better: embed top-token descriptions with a sentence encoder, cluster the embeddings to get K semantic categories (e.g., "punctuation", "named entities", "syntax", "math")
2. Use **SimpleARTMAP** with:
   - A-side: FuzzyART clustering the decoder weight vectors (geometry)
   - B-side: the semantic category labels (meaning)
   - Match tracking ensures geometric clusters are consistent with semantic labels
3. When a geometric cluster tries to contain features with conflicting labels, match tracking raises vigilance, forcing finer-grained splits until geometric and semantic categories align

**What this adds**: Semantically grounded taxonomy. The hierarchy isn't just "geometrically similar features" but "geometrically similar features that also mean similar things." Match tracking automatically discovers the right granularity for semantic coherence — this is ART's unique contribution vs. k-means or agglomerative clustering.

**artlib class**: `SimpleARTMAP(module_a=FuzzyART(...))` with `.fit(X_decoder, y_semantic_labels)`

---

### Enhancement 5: Temporal Stability Analysis via Incremental Fit (Phase 2)

**Grossberg parallel**: Stability-plasticity dilemma — the core problem ART was designed to solve. Also: episodic memory and contextually cued recall.

**Motivation**: Already stubbed in the notebook. EleutherAI's finding that SAE features are unstable across seeds makes this more urgent — we need to distinguish stable feature structure from noise.

**Approach**:
1. Load Pythia-70M SAEs at multiple training checkpoints (EleutherAI provides these)
2. At checkpoint 0: fit ART taxonomy from scratch
3. At each subsequent checkpoint: use `partial_fit()` to incrementally update
4. Track per-category metrics:
   - **Resonant categories**: existing categories that keep receiving new members (stable features)
   - **Growing categories**: categories whose membership increases (features becoming more prominent)
   - **Dormant categories**: categories that stop receiving members (features that die during training)
   - **Novel categories**: newly created categories (features emerging during training)
   - **Vigilance violations**: how often match tracking is triggered (feature space turbulence)

**What this adds**: A developmental timeline of feature emergence. This directly tests Grossberg's central claim that ART handles non-stationary data without catastrophic forgetting. If the taxonomy remains coherent across checkpoints while capturing genuine changes, it validates ART as a tool for tracking LLM training dynamics.

**artlib method**: `model.partial_fit(X_new)` — already supported by both artlib FuzzyART and GPUFuzzyART

---

### Enhancement 6: Multi-Signal Feature Fusion via FusionART

**Grossberg parallel**: CogEM/MOTIVATOR — cognitive-emotional resonance linking object categories to value categories across multiple brain systems operating in parallel.

**Motivation**: SAE features have multiple informative signals beyond decoder weight geometry:
- The decoder weight vector itself (what direction in activation space)
- Activation statistics (how often/strongly the feature fires)
- Token-level semantics (what contexts trigger it)
- Cross-layer persistence (does it appear in adjacent layers)

Currently we only use the first signal.

**Approach**:
1. Define multiple channels per feature:
   - **Channel 1 (Geometry)**: PCA-reduced decoder weight vector (current approach)
   - **Channel 2 (Behavior)**: Activation statistics vector (mean activation, frequency, max activation, entropy of activations across a corpus)
   - **Channel 3 (Semantics)**: Sentence-encoder embedding of top-activating tokens/contexts
2. Use **FusionART** with per-channel gamma weights controlling the relative importance of each signal
3. Each channel gets its own ART module, but the fused activation and match functions enforce cross-channel consistency

**What this adds**: Richer, multi-modal feature categories. A category isn't just "features that point in similar directions" but "features that point similarly, fire similarly, and mean similar things." The gamma weights allow ablation studies: which signal matters most for taxonomy quality?

**artlib class**: `FusionART(modules=[FuzzyART(...), FuzzyART(...), FuzzyART(...)], gamma_values=[...], channel_dims=[...])`

---

### Enhancement 7: DualVigilanceART for Complex Feature Boundaries

**Grossberg parallel**: Complex category shapes in visual object recognition — a single prototype cannot capture the full diversity of views/instances.

**Motivation**: SAE feature clusters in decoder weight space are often non-convex. FuzzyART produces hyper-box categories, which may split natural clusters or merge unrelated ones when shapes are irregular.

**Approach**:
1. Wrap FuzzyART in **DualVigilanceART** with:
   - Upper vigilance `rho_upper`: controls the fine-grained base categories (tight clusters)
   - Lower vigilance `rho_lower`: controls how base categories are merged into abstract categories (loose groupings)
2. The result: abstract categories with arbitrary shapes, composed of multiple hyper-box sub-categories

**What this adds**: Better cluster quality for irregularly shaped feature families. This is particularly important for features that form manifolds or elongated structures in the decoder weight space (as suggested by the "Geometry of Concepts" work showing parallelogram and crystal structures).

**artlib class**: `DualVigilanceART(base_module=FuzzyART(...), rho_lower_bound=0.5)`

---

## Recommended Implementation Order

| Priority | Enhancement | Complexity | New Insight |
|----------|------------|------------|-------------|
| 1 | **E4: Supervised taxonomy (SimpleARTMAP)** | Low | Semantically grounded clusters |
| 2 | **E1: Co-activation topology (TopoART)** | Medium | Feature interaction graph |
| 3 | **E3: Feature-token biclustering (BARTMAP)** | Medium | Bidirectional feature-token maps |
| 4 | **E5: Temporal analysis (partial_fit)** | Medium | Feature developmental timeline |
| 5 | **E7: Dual vigilance (DualVigilanceART)** | Low | Better cluster shapes |
| 6 | **E2: Cross-layer flow (ARTMAP)** | High | Feature evolution across layers |
| 7 | **E6: Multi-signal fusion (FusionART)** | High | Multi-modal feature categories |

E4 first because it requires minimal new infrastructure (just auto-labels + SimpleARTMAP) but immediately validates whether ART's match tracking produces better semantic coherence than unsupervised clustering alone.

---

## Key References

### SAE Interpretability
- Chanin et al. (2024). "A is for Absorption." NeurIPS 2024. [arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- Gao et al. (2024). "Scaling and Evaluating Sparse Autoencoders." ICLR 2025. [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- Marks et al. (2024). "Sparse Feature Circuits." ICLR 2025. [arXiv:2403.19647](https://arxiv.org/abs/2403.19647)
- Li, Michaud et al. (2024). "The Geometry of Concepts." Entropy 2025. [arXiv:2410.19750](https://arxiv.org/abs/2410.19750)
- (2025). "Sparse Feature Coactivation Reveals Causal Semantic Modules in LLMs." [arXiv:2506.18141](https://arxiv.org/abs/2506.18141)
- (2025). "Matryoshka Sparse Autoencoders." ICML 2025. [arXiv:2503.17547](https://arxiv.org/abs/2503.17547)
- (2026). "From Atoms to Trees: HSAE." [arXiv:2602.11881](https://arxiv.org/abs/2602.11881)
- EleutherAI (2024). "SAEs Trained on Same Data Don't Learn Same Features." [Blog](https://blog.eleuther.ai/sae_seed_similarity/)
- Anthropic (2024). "Sparse Crosscoders." [Transformer Circuits](https://transformer-circuits.pub/2024/crosscoders/index.html)
- Anthropic (2025). "Circuit Tracing." [Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- Karvonen et al. (2025). "SAEBench." ICML 2025. [Neuronpedia](https://www.neuronpedia.org/sae-bench/info)

### ART Theory & Tools
- Grossberg (2025). "Neural network models of autonomous adaptive intelligence and AGI." Frontiers in Systems Neuroscience. [DOI:10.3389/fnsys.2025.1630151](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1630151/full)
- Grossberg (2020). "A Path Toward Explainable AI." Frontiers in Neurorobotics. [DOI:10.3389/fnbot.2020.00036](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2020.00036/full)
- Melton et al. (2025). "Deep ARTMAP." [arXiv:2503.07641](https://arxiv.org/abs/2503.07641)
- Petrenko et al. (2025). "DeepART." Neural Networks. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608025004605)
- Da Silva et al. (2019). "A Survey of ART Neural Network Models." Neural Networks. [arXiv:1905.11437](https://arxiv.org/abs/1905.11437)
- Melton (2025). "artlib: A Python package for ART." JOSS. [DOI:10.21105/joss.07764](https://joss.theoj.org/papers/10.21105/joss.07764.pdf)
