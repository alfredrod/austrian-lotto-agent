"""Tests for the probability models."""
import numpy as np
import pytest

from models import (MODEL_REGISTRY, BayesianFrequencyModel, RecencyWeightedModel, UniformRandomBaseline,
                    default_models, get_model, inclusion_from_weights)


def random_draws(n: int, seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(np.arange(1, 46), 6, replace=False).tolist()) for _ in range(n)]


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_fit_predict_suggest(name):
    m = get_model(name).fit(random_draws(300))
    p = m.predict_probabilities()
    assert p.shape == (45,) and np.isclose(p.sum(), 6) and (p > 0).all() and (p <= 1).all()
    s = m.suggest_numbers(np.random.default_rng(1))
    assert len(s) == 6 and len(set(s)) == 6 and all(1 <= x <= 45 for x in s)


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        BayesianFrequencyModel().predict_probabilities()


def test_uniform_baseline():
    m = UniformRandomBaseline().fit(random_draws(10))
    assert np.allclose(m.predict_probabilities(), 6 / 45)
    s = m.suggest_numbers()
    assert len(set(s)) == 6


def test_bayesian_shrinkage():
    draws = [[1, 2, 3, 4, 5, 6]] * 50
    weak = BayesianFrequencyModel(alpha=1).fit(draws).predict_probabilities()
    strong = BayesianFrequencyModel(alpha=10000).fit(draws).predict_probabilities()
    assert weak[0] > strong[0] > 6 / 45
    assert abs(strong[0] - 6 / 45) < 0.01


def test_recency_prefers_recent():
    draws = [[1, 2, 3, 4, 5, 6]] * 100 + [[40, 41, 42, 43, 44, 45]] * 100
    p = RecencyWeightedModel(half_life=10, prior_strength=1).fit(draws).predict_probabilities()
    assert p[44] > p[0]


def test_inclusion_capping():
    w = np.full(45, 1e-6)
    w[0] = 1.0
    p = inclusion_from_weights(w / w.sum())
    assert np.isclose(p.sum(), 6) and p.max() <= 1 + 1e-9


def test_empty_history_ok():
    for m in default_models():
        assert np.isclose(m.fit([]).predict_probabilities().sum(), 6)
