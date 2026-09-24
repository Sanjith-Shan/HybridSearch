"""Click models for simulated users.

Every click this module produces comes from a *simulated* user. Relevance comes from
real graded TREC DL judgments; the behaviour on top of it is a textbook click model.
Nothing here is user traffic.

Models (all vectorised over sessions, seeded through an explicit ``numpy.random.Generator``):

* **PBM**, the position-based model (Richardson et al. 2007; Craswell et al. 2008):
  ``P(click at r) = eta_r * alpha(grade)``, examination independent of everything else.
  ``eta_r = (1 / r) ** severity`` as in Joachims, Swaminathan & Schnabel (WSDM 2017).
* **Cascade** in the generalised form used by Hofmann, Whiteson & de Rijke (2011, 2013)
  (the dependent click model of Guo et al. 2009): the user scans top-down, examines
  every result until they stop, clicks with ``alpha(grade)``, and after a click stops
  (is satisfied) with ``sigma(grade)``. With ``sigma == 1`` it is Craswell's pure cascade.
* **DBN**, the dynamic Bayesian network model (Chapelle & Zhang, WWW 2009):
  ``E_1 = 1``; ``C_r = E_r * A_r`` with ``P(A_r)=alpha``; ``S_r = C_r * Bernoulli(sigma)``;
  ``E_{r+1} = E_r * (1 - S_r) * Bernoulli(gamma)``. The Cascade model above is DBN with
  ``gamma = 1``.

Relevance → behaviour mapping. The user profiles are the standard ones from the online
LTR literature (Hofmann et al. 2013 for 3 grades; the 5-grade extension used by Schuth et
al. 2016 and Oosterhuis & de Rijke 2018 on MSLR-WEB10K), tabulated below for grades
0..4. TREC DL has grades 0..3 where grade 1 ("related") is *below* the binary-relevance
threshold (TREC DL counts grade >= 2 as relevant). We therefore map DL grades
``{0, 1, 2, 3}`` onto table rows ``{0, 1, 3, 4}``: "related" gets the weak-relevance row,
"highly relevant" and "perfectly relevant" get the two top rows. Unjudged passages are
treated as grade 0 (the trec_eval convention). The mapping is a parameter
(``DL_TO_TABLE``) so it can be changed and re-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 5-level tables, rows = table grade 0..4.
PROFILES: dict[str, dict[str, list[float]]] = {
    "perfect": {"click": [0.0, 0.2, 0.4, 0.8, 1.0], "stop": [0.0, 0.0, 0.0, 0.0, 0.0]},
    "navigational": {"click": [0.05, 0.3, 0.5, 0.7, 0.95], "stop": [0.2, 0.3, 0.5, 0.7, 0.9]},
    "informational": {"click": [0.4, 0.6, 0.7, 0.8, 0.9], "stop": [0.1, 0.2, 0.3, 0.4, 0.5]},
}

# TREC DL grade -> row of the 5-level table (see module docstring).
DL_TO_TABLE = np.array([0, 1, 3, 4], dtype=np.int64)

NO_DOC = -1  # grade value marking an empty slot (list shorter than the page)


def dl_profile_arrays(profile: str, mapping: np.ndarray = DL_TO_TABLE) -> tuple[np.ndarray, np.ndarray]:
    """(alpha, sigma) indexed by TREC DL grade 0..3."""
    p = PROFILES[profile]
    return np.asarray(p["click"], float)[mapping], np.asarray(p["stop"], float)[mapping]


def position_bias(k: int, severity: float = 1.0) -> np.ndarray:
    """eta_r = (1/r)^severity for r = 1..k (Joachims et al. 2017)."""
    return (1.0 / np.arange(1, k + 1, dtype=float)) ** severity


@dataclass
class ClickResult:
    clicks: np.ndarray      # (n, K) bool
    satisfied: np.ndarray   # (n, K) bool, satisfied click (drives long dwell)
    examined: np.ndarray    # (n, K) bool, latent examination (for tests / diagnostics)


def _lookup(table: np.ndarray, grades: np.ndarray) -> np.ndarray:
    g = np.asarray(grades)
    out = np.zeros(g.shape, dtype=float)
    ok = g >= 0
    out[ok] = table[np.clip(g[ok], 0, len(table) - 1)]
    return out


@dataclass
class ClickModel:
    """Base: ``alpha``/``sigma`` indexed by DL grade."""

    name: str
    alpha: np.ndarray
    sigma: np.ndarray

    def simulate(self, grades: np.ndarray, rng: np.random.Generator) -> ClickResult:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "alpha_by_dl_grade": self.alpha.tolist(),
                "sigma_by_dl_grade": self.sigma.tolist()}


@dataclass
class PBM(ClickModel):
    eta: np.ndarray = field(default_factory=lambda: position_bias(10))

    def simulate(self, grades, rng):
        grades = np.asarray(grades)
        n, k = grades.shape
        eta = self.eta[:k]
        examined = rng.random((n, k)) < eta
        attracted = rng.random((n, k)) < _lookup(self.alpha, grades)
        clicks = examined & attracted
        # PBM has no satisfaction variable; for the dwell signal only, a click is
        # "satisfied" with probability sigma(grade).
        satisfied = clicks & (rng.random((n, k)) < _lookup(self.sigma, grades))
        return ClickResult(clicks, satisfied, examined)

    def describe(self):
        d = super().describe()
        d["eta"] = self.eta.tolist()
        return d


@dataclass
class DBN(ClickModel):
    """DBN with continuation ``gamma``. ``gamma = 1`` gives the Cascade/DCM model."""

    gamma: float = 0.9

    def simulate(self, grades, rng):
        grades = np.asarray(grades)
        n, k = grades.shape
        a = _lookup(self.alpha, grades)
        s = _lookup(self.sigma, grades)
        u_click = rng.random((n, k))
        u_sat = rng.random((n, k))
        u_cont = rng.random((n, k))
        clicks = np.zeros((n, k), bool)
        sat = np.zeros((n, k), bool)
        examined = np.zeros((n, k), bool)
        e = np.ones(n, bool)
        for r in range(k):
            examined[:, r] = e
            c = e & (u_click[:, r] < a[:, r])
            sr = c & (u_sat[:, r] < s[:, r])
            clicks[:, r] = c
            sat[:, r] = sr
            e = e & ~sr & (u_cont[:, r] < self.gamma)
        return ClickResult(clicks, sat, examined)

    def describe(self):
        d = super().describe()
        d["gamma"] = self.gamma
        return d


def make_model(kind: str, profile: str, k: int = 10, severity: float = 1.0,
               gamma: float = 0.9) -> ClickModel:
    """``kind`` in {pbm, cascade, dbn}; ``profile`` in PROFILES."""
    alpha, sigma = dl_profile_arrays(profile)
    name = f"{kind}-{profile}"
    if kind == "pbm":
        return PBM(name, alpha, sigma, eta=position_bias(k, severity))
    if kind == "cascade":
        return DBN(name, alpha, sigma, gamma=1.0)
    if kind == "dbn":
        return DBN(name, alpha, sigma, gamma=gamma)
    raise ValueError(f"unknown click model {kind!r}")


ALL_MODELS = [(k, p) for k in ("pbm", "cascade", "dbn") for p in ("perfect", "navigational", "informational")]


def sample_dwell_ms(satisfied: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dwell times for clicks: satisfied clicks are long (lognormal, median 60 s),
    unsatisfied short (median 8 s). 30 s is the usual satisfied-click threshold
    (Fox et al. 2005), so the two populations straddle it. Simulated, not measured."""
    med = np.where(satisfied, 60_000.0, 8_000.0)
    return np.round(med * np.exp(0.6 * rng.standard_normal(satisfied.shape))).astype(np.int64)
