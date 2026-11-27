"""Probability models for Lotto 6 aus 45.

Every model outputs per-number *inclusion probabilities* for the next draw (length-45 vector summing to 6,
each <= 1). Under a fair draw each number has inclusion probability 6/45 ~= 0.1333. Because the six numbers
are drawn without replacement, counts per draw are 0/1 per number; models therefore work with normalised
selection weights and convert them to inclusion probabilities with a capped k*w transform, and suggestions
are sampled sequentially without replacement.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge

N_NUMBERS = 45
K_PICK = 6
UNIFORM_P = K_PICK / N_NUMBERS

DrawLike = Sequence[int]


def to_indicator_matrix(draws: Sequence[Any]) -> np.ndarray:
    """T x 45 0/1 matrix from draws (Draw objects or sequences of 6 ints)."""
    mat = np.zeros((len(draws), N_NUMBERS), dtype=np.float64)
    for t, d in enumerate(draws):
        nums = getattr(d, "numbers", d)
        mat[t, np.asarray(nums, dtype=int) - 1] = 1.0
    return mat


def inclusion_from_weights(weights: np.ndarray, k: int = K_PICK) -> np.ndarray:
    """Convert selection weights (sum 1) to inclusion probabilities (sum k, each <= 1) by iterative capping."""
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, None)
    w = w / w.sum()
    p = k * w
    for _ in range(N_NUMBERS):
        over = p > 1.0
        if not over.any():
            break
        free = ~over & (p < 1.0)
        excess = (p[over] - 1.0).sum()
        p[over] = 1.0
        if not free.any():
            break
        p[free] += excess * p[free] / p[free].sum()
    return p


class BaseModel(ABC):
    """Abstract model: fit on chronological past draws, output inclusion probabilities, suggest 6 numbers."""

    name: str = "base"

    def __init__(self) -> None:
        self._weights: Optional[np.ndarray] = None
        self.n_train: int = 0

    @property
    def params(self) -> dict[str, Any]:
        """Hyper-parameters (for logging)."""
        return {}

    @abstractmethod
    def _fit_weights(self, X: np.ndarray) -> np.ndarray:
        """Return selection weights (length 45) from a T x 45 indicator matrix."""

    def fit(self, draws: Sequence[Any]) -> "BaseModel":
        """Fit on draws ordered oldest -> newest. Only data passed here may be used."""
        X = to_indicator_matrix(draws) if len(draws) else np.zeros((0, N_NUMBERS))
        self.n_train = X.shape[0]
        w = self._fit_weights(X)
        if w.shape != (N_NUMBERS,) or not np.all(np.isfinite(w)):
            raise ValueError(f"{self.name}: invalid weights")
        self._weights = w / w.sum()
        return self

    def predict_probabilities(self) -> np.ndarray:
        """Inclusion probabilities for numbers 1..45 in the next draw (sum = 6)."""
        if self._weights is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return inclusion_from_weights(self._weights)

    def suggest_numbers(self, rng: Optional[np.random.Generator] = None) -> list[int]:
        """Sample 6 distinct numbers sequentially without replacement, proportional to model weights."""
        if self._weights is None:
            raise RuntimeError(f"{self.name} must be fitted before suggesting")
        rng = rng or np.random.default_rng()
        picks = rng.choice(np.arange(1, N_NUMBERS + 1), size=K_PICK, replace=False, p=self._weights)
        return sorted(int(x) for x in picks)


class UniformRandomBaseline(BaseModel):
    """Fair-lottery baseline: every number equally likely; suggestions from the OS entropy source."""

    name = "uniform_baseline"

    def _fit_weights(self, X: np.ndarray) -> np.ndarray:
        return np.full(N_NUMBERS, 1.0 / N_NUMBERS)

    def suggest_numbers(self, rng: Optional[np.random.Generator] = None) -> list[int]:
        """Truly random 6 of 45 (SystemRandom) unless a seeded rng is given for reproducibility."""
        if self._weights is None:
            self.fit([])
        if rng is not None:
            return super().suggest_numbers(rng)
        return sorted(random.SystemRandom().sample(range(1, N_NUMBERS + 1), K_PICK))


class BayesianFrequencyModel(BaseModel):
    """Dirichlet prior (symmetric, strength alpha per number) on selection weights, updated with counts.

    Posterior mean w_i = (alpha + c_i) / (45*alpha + 6T): shrinks observed frequencies toward uniform.
    """

    name = "bayesian_frequency"

    def __init__(self, alpha: float = 100.0) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha

    @property
    def params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}

    def _fit_weights(self, X: np.ndarray) -> np.ndarray:
        return self.alpha + X.sum(axis=0)


class RecencyWeightedModel(BaseModel):
    """Exponentially decayed counts (half-life in draws) plus a uniform pseudo-count."""

    name = "recency_weighted"

    def __init__(self, half_life: float = 100.0, prior_strength: float = 20.0) -> None:
        super().__init__()
        if half_life <= 0 or prior_strength < 0:
            raise ValueError("half_life must be > 0 and prior_strength >= 0")
        self.half_life = half_life
        self.prior_strength = prior_strength

    @property
    def params(self) -> dict[str, Any]:
        return {"half_life": self.half_life, "prior_strength": self.prior_strength}

    def _fit_weights(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        decay = 0.5 ** (np.arange(T)[::-1] / self.half_life)
        return self.prior_strength + 1e-9 + decay @ X


class RegularizedHistoricalModel(BaseModel):
    """Ridge regression predicting next-draw inclusion from history features of each number.

    Features per (draw t, number i), computed from draws strictly before t: overall frequency, frequency in
    the last 10 and 50 draws, and draws since last appearance (scaled). Strong L2 regularisation keeps the
    prediction close to the mean rate when features carry no signal.
    """

    name = "ridge_historical"

    def __init__(self, ridge_alpha: float = 1000.0, max_train_draws: int = 1000, min_history: int = 50) -> None:
        super().__init__()
        self.ridge_alpha = ridge_alpha
        self.max_train_draws = max_train_draws
        self.min_history = min_history

    @property
    def params(self) -> dict[str, Any]:
        return {"ridge_alpha": self.ridge_alpha, "max_train_draws": self.max_train_draws,
                "min_history": self.min_history}

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        """Features for predicting row t (t = 0..T) using rows < t. Returns (T+1) x 45 x 4."""
        T = X.shape[0]
        cs = np.vstack([np.zeros(N_NUMBERS), np.cumsum(X, axis=0)])  # cs[t] = counts in rows < t
        t_idx = np.arange(T + 1)[:, None]

        def window(w: int) -> np.ndarray:
            lo = np.maximum(t_idx - w, 0)
            return (cs - cs[lo.ravel()]) / np.maximum(np.minimum(t_idx, w), 1)

        overall = cs / np.maximum(t_idx, 1)
        seen_idx = np.where(X > 0, np.arange(T)[:, None], -1)
        last_seen = np.vstack([np.full(N_NUMBERS, -1), np.maximum.accumulate(seen_idx, axis=0)]) if T else \
            np.full((1, N_NUMBERS), -1)
        gap = np.where(last_seen >= 0, t_idx - 1 - last_seen, t_idx) / 20.0
        return np.stack([overall, window(10), window(50), gap], axis=-1)

    def _fit_weights(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        if T <= self.min_history:
            return np.full(N_NUMBERS, 1.0 / N_NUMBERS)
        F = self._features(X)
        start = max(self.min_history, T - self.max_train_draws)
        Xtr = F[start:T].reshape(-1, F.shape[-1])
        ytr = X[start:T].reshape(-1)
        reg = Ridge(alpha=self.ridge_alpha).fit(Xtr, ytr)
        pred = reg.predict(F[T])
        return np.clip(pred, 1e-3, None)


MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    m.name: m for m in (UniformRandomBaseline, BayesianFrequencyModel, RecencyWeightedModel, RegularizedHistoricalModel)
}


def get_model(name: str, **params: Any) -> BaseModel:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


def default_models() -> list[BaseModel]:
    """One instance of every model with default hyper-parameters."""
    return [cls() for cls in MODEL_REGISTRY.values()]
