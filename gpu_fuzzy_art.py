"""GPU-accelerated Fuzzy ART and SMART implementations using PyTorch.

Provides vectorized activation/match computation across all clusters on GPU,
while maintaining full API compatibility with the notebook pipeline.
"""

import numpy as np
import torch
from typing import Optional


class GPUFuzzyART:
    """GPU-accelerated Fuzzy ART clustering using PyTorch.

    Standalone implementation that vectorizes activation and match computation
    across all clusters on GPU. Compatible with the artlib API surface used
    by the notebook (prepare_data, fit, predict, labels_, n_clusters, params).

    Parameters
    ----------
    rho : float
        Vigilance parameter in [0, 1]. Higher values produce more clusters.
    alpha : float
        Choice parameter (bias toward small clusters). Typical: 0.01.
    beta : float
        Learning rate in [0, 1]. 1.0 = fast learning (winner-take-all).
    epsilon : float
        Match tracking increment for MT+.
    device : str or torch.device, optional
        Compute device. Auto-detects CUDA if available.
    dtype : torch.dtype
        Tensor data type.
    """

    def __init__(
        self,
        rho: float = 0.7,
        alpha: float = 0.01,
        beta: float = 1.0,
        epsilon: float = 1e-6,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.params = {"rho": rho, "alpha": alpha, "beta": beta}
        self._epsilon = epsilon
        self._dtype = dtype

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Weight buffer (pre-allocated, grows by doubling)
        self._w: Optional[torch.Tensor] = None  # (capacity, 2*dim)
        self._num_categories: int = 0
        self._w_norms: Optional[torch.Tensor] = None  # cached L1 norms of weights

        # Normalization bounds
        self.d_min_: Optional[np.ndarray] = None
        self.d_max_: Optional[np.ndarray] = None

        # Fit results
        self.labels_: np.ndarray = np.zeros(0, dtype=int)
        self._dim_original: Optional[int] = None  # original dim (before complement coding)

    # ------------------------------------------------------------------
    # Properties matching artlib API
    # ------------------------------------------------------------------

    @property
    def n_clusters(self) -> int:
        return self._num_categories

    # ------------------------------------------------------------------
    # Data preparation (replicates artlib normalize + complement_code)
    # ------------------------------------------------------------------

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize to [0,1] and apply complement coding.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, dim)
            Raw input data.

        Returns
        -------
        np.ndarray, shape (n_samples, 2*dim)
            Complement-coded data ready for fit/predict.
        """
        X = np.array(X, dtype=np.float64)

        if self.d_min_ is None:
            self.d_min_ = X.min(axis=0)
        if self.d_max_ is None:
            self.d_max_ = X.max(axis=0)

        range_vals = self.d_max_ - self.d_min_
        mask = range_vals == 0

        normalized = np.zeros_like(X, dtype=np.float64)
        normalized[:, ~mask] = (X[:, ~mask] - self.d_min_[~mask]) / range_vals[~mask]

        # Complement coding: [x, 1-x]
        cc_data = np.hstack([normalized, 1.0 - normalized])
        return cc_data

    # ------------------------------------------------------------------
    # Weight buffer management
    # ------------------------------------------------------------------

    def _init_weights(self, dim2: int, initial_capacity: int = 1000):
        """Allocate the weight buffer."""
        self._w = torch.empty(
            initial_capacity, dim2, device=self._device, dtype=self._dtype
        )
        self._w_norms = torch.empty(
            initial_capacity, device=self._device, dtype=self._dtype
        )
        self._num_categories = 0

    def _ensure_capacity(self, needed: int):
        """Double buffer capacity if it would overflow."""
        if self._w is not None and needed <= self._w.shape[0]:
            return
        new_cap = max(1000, self._w.shape[0] * 2) if self._w is not None else 1000
        while new_cap < needed:
            new_cap *= 2

        new_w = torch.empty(new_cap, self._w.shape[1], device=self._device, dtype=self._dtype)
        new_norms = torch.empty(new_cap, device=self._device, dtype=self._dtype)

        if self._w is not None and self._num_categories > 0:
            new_w[: self._num_categories] = self._w[: self._num_categories]
            new_norms[: self._num_categories] = self._w_norms[: self._num_categories]

        self._w = new_w
        self._w_norms = new_norms

    @property
    def _weights(self) -> torch.Tensor:
        """View of active weights only."""
        return self._w[: self._num_categories]

    @property
    def _weight_norms(self) -> torch.Tensor:
        """View of active weight L1 norms."""
        return self._w_norms[: self._num_categories]

    def _add_category(self, w_new: torch.Tensor):
        """Add a new category with weight vector w_new (1D tensor)."""
        idx = self._num_categories
        self._ensure_capacity(idx + 1)
        self._w[idx] = w_new
        self._w_norms[idx] = w_new.abs().sum()
        self._num_categories += 1

    def _update_category(self, idx: int, w_new: torch.Tensor):
        """Overwrite weight and cached norm for category idx."""
        self._w[idx] = w_new
        self._w_norms[idx] = w_new.abs().sum()

    # ------------------------------------------------------------------
    # Core: single-sample step
    # ------------------------------------------------------------------

    def _step_fit(self, x: torch.Tensor) -> int:
        """Fit a single sample and return its cluster label.

        Parameters
        ----------
        x : torch.Tensor, shape (2*dim,)
            Single complement-coded sample (already on device).

        Returns
        -------
        int
            Assigned cluster index.
        """
        if self._num_categories == 0:
            self._add_category(x.clone())
            return 0

        rho = self.params["rho"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        # Vectorized activation & match across ALL categories
        w = self._weights  # (K, dim2)
        x_and_w = torch.minimum(x.unsqueeze(0), w)  # (K, dim2)
        l1_norms = x_and_w.sum(dim=1)  # (K,)

        T = l1_norms / (alpha + self._weight_norms)
        M = l1_norms / self._dim_original

        # Search: iterate by activation descending
        sorted_indices = torch.argsort(T, descending=True)

        for si in sorted_indices:
            j = si.item()
            if M[j].item() >= rho:
                w_new = beta * x_and_w[j] + (1.0 - beta) * w[j]
                self._update_category(j, w_new)
                return j

        # No match — create new category
        c_new = self._num_categories
        self._add_category(x.clone())
        return c_new

    # ------------------------------------------------------------------
    # Core: fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, max_iter: int = 1) -> "GPUFuzzyART":
        """Fit the model to complement-coded data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 2*dim)
            Complement-coded data (output of prepare_data).
        max_iter : int
            Number of passes over the data.

        Returns
        -------
        self
        """
        n_samples, dim2 = X.shape
        self._dim_original = dim2 / 2.0

        X_t = torch.as_tensor(X, device=self._device, dtype=self._dtype)

        self._init_weights(dim2)
        self.labels_ = np.zeros(n_samples, dtype=int)

        for _epoch in range(max_iter):
            for i in range(n_samples):
                self.labels_[i] = self._step_fit(X_t[i])

        return self

    # ------------------------------------------------------------------
    # Core: partial_fit (online / incremental)
    # ------------------------------------------------------------------

    def partial_fit(self, X: np.ndarray) -> "GPUFuzzyART":
        """Incrementally fit one or more samples.

        Call this repeatedly to train sample-by-sample (e.g. inside a
        tqdm loop).  The first call initialises internal state; subsequent
        calls extend ``labels_``.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 2*dim)
            One or more complement-coded samples.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dim2 = X.shape[1]

        # First-call initialisation
        if self._w is None:
            self._dim_original = dim2 / 2.0
            self._init_weights(dim2)
            self.labels_ = np.zeros(0, dtype=int)

        X_t = torch.as_tensor(X, device=self._device, dtype=self._dtype)
        new_labels = np.empty(X_t.shape[0], dtype=int)

        for i in range(X_t.shape[0]):
            new_labels[i] = self._step_fit(X_t[i])

        self.labels_ = np.concatenate([self.labels_, new_labels])
        return self

    # ------------------------------------------------------------------
    # Core: predict (fully batched)
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for complement-coded data.

        Uses fully batched GPU computation, chunked to avoid OOM.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 2*dim)
            Complement-coded data (output of prepare_data).

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted cluster labels.
        """
        if self._num_categories == 0:
            raise RuntimeError("Model has not been fitted yet.")

        n_samples = X.shape[0]
        X_t = torch.as_tensor(X, device=self._device, dtype=self._dtype)
        w = self._weights  # (K, dim2)
        w_norms = self._weight_norms  # (K,)
        alpha = self.params["alpha"]

        chunk_size = 4096
        labels = np.empty(n_samples, dtype=int)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            batch = X_t[start:end]  # (B, dim2)

            # Vectorize across both samples AND categories
            # batch: (B, 1, dim2), w: (1, K, dim2)
            x_and_w = torch.minimum(batch.unsqueeze(1), w.unsqueeze(0))  # (B, K, dim2)
            l1_norms = x_and_w.sum(dim=2)  # (B, K)

            # Activation: pick cluster with highest T
            T = l1_norms / (alpha + w_norms.unsqueeze(0))  # (B, K)
            labels[start:end] = T.argmax(dim=1).cpu().numpy()

        return labels

    # ------------------------------------------------------------------
    # Constrained fit (for GPUSMART hierarchical levels)
    # ------------------------------------------------------------------

    def _step_fit_constrained(self, x: torch.Tensor, p_label: int) -> int:
        """Fit a single sample with parent-label constraint.

        Parameters
        ----------
        x : torch.Tensor, shape (2*dim,)
            Single complement-coded sample (already on device).
        p_label : int
            Parent-level cluster label for this sample.

        Returns
        -------
        int
            Assigned cluster index.
        """
        if self._num_categories == 0:
            self._add_category(x.clone())
            self._cluster_parent[0] = p_label
            return 0

        rho = self.params["rho"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        epsilon = self._epsilon

        w = self._weights
        x_and_w = torch.minimum(x.unsqueeze(0), w)
        l1_norms = x_and_w.sum(dim=1)
        T = l1_norms / (alpha + self._weight_norms)
        M = l1_norms / self._dim_original

        current_rho = rho
        sorted_indices = torch.argsort(T, descending=True)

        for si in sorted_indices:
            j = si.item()
            m_val = M[j].item()

            if m_val < current_rho:
                # Match criterion failed — skip, no rho change
                continue

            # Match criterion passed; check parent constraint
            if j in self._cluster_parent and self._cluster_parent[j] != p_label:
                # Constraint violation — match tracking (MT+)
                current_rho = m_val + epsilon
                continue

            # Resonance: update weight
            w_new = beta * x_and_w[j] + (1.0 - beta) * w[j]
            self._update_category(j, w_new)
            self._cluster_parent[j] = p_label
            return j

        # No match — create new category
        c_new = self._num_categories
        self._add_category(x.clone())
        self._cluster_parent[c_new] = p_label
        return c_new

    def _fit_constrained(
        self,
        X: np.ndarray,
        parent_labels: np.ndarray,
        max_iter: int = 1,
    ) -> "GPUFuzzyART":
        """Fit with hierarchical constraint: each cluster may only contain
        samples from a single parent cluster.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 2*dim)
            Complement-coded data.
        parent_labels : np.ndarray, shape (n_samples,)
            Cluster labels from the coarser (parent) level.
        max_iter : int
            Number of passes.

        Returns
        -------
        self
        """
        n_samples, dim2 = X.shape
        self._dim_original = dim2 / 2.0

        X_t = torch.as_tensor(X, device=self._device, dtype=self._dtype)

        self._init_weights(dim2)
        self.labels_ = np.zeros(n_samples, dtype=int)
        self._cluster_parent: dict[int, int] = {}

        for _epoch in range(max_iter):
            for i in range(n_samples):
                self.labels_[i] = self._step_fit_constrained(
                    X_t[i], int(parent_labels[i])
                )

        return self


class GPUSMART:
    """GPU-accelerated SMART hierarchical clustering.

    Creates a GPUFuzzyART instance per vigilance level and fits them
    with hierarchical constraints so each level subdivides the clusters
    from the previous level.

    Parameters
    ----------
    rho_values : list[float]
        Monotonically increasing vigilance values, one per hierarchy level.
    alpha : float
        Choice parameter.
    beta : float
        Learning rate.
    epsilon : float
        Match tracking increment.
    device : str or torch.device, optional
        Compute device.
    dtype : torch.dtype
        Tensor data type.
    """

    def __init__(
        self,
        rho_values: list,
        alpha: float = 0.01,
        beta: float = 1.0,
        epsilon: float = 1e-6,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        assert all(
            rho_values[i] < rho_values[i + 1] for i in range(len(rho_values) - 1)
        ), "rho_values must be monotonically increasing"

        self.rho_values = list(rho_values)
        self.modules: list[GPUFuzzyART] = []
        for rho in rho_values:
            self.modules.append(
                GPUFuzzyART(
                    rho=rho,
                    alpha=alpha,
                    beta=beta,
                    epsilon=epsilon,
                    device=device,
                    dtype=dtype,
                )
            )

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data using the first module, then share normalization bounds.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, dim)
            Raw input data.

        Returns
        -------
        np.ndarray, shape (n_samples, 2*dim)
            Complement-coded data.
        """
        result = self.modules[0].prepare_data(X)
        # Copy normalization bounds to all other modules
        for m in self.modules[1:]:
            m.d_min_ = self.modules[0].d_min_.copy()
            m.d_max_ = self.modules[0].d_max_.copy()
        return result

    def fit(self, X: np.ndarray, max_iter: int = 1) -> "GPUSMART":
        """Fit the hierarchical model.

        Level 0 is unconstrained. Each subsequent level is constrained so that
        every cluster contains samples from only one parent-level cluster.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 2*dim)
            Complement-coded data (output of prepare_data).
        max_iter : int
            Number of passes per level.

        Returns
        -------
        self
        """
        # Level 0: unconstrained
        self.modules[0].fit(X, max_iter=max_iter)

        # Levels 1+: constrained by previous level's labels
        for k in range(1, len(self.modules)):
            self.modules[k]._fit_constrained(
                X,
                parent_labels=self.modules[k - 1].labels_,
                max_iter=max_iter,
            )

        return self
