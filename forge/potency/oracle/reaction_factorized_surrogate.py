"""A role-decomposed Bayesian surrogate over Ugi lipids.

Why this model and not something larger. Under component-family-aware cross-validation the
production D-MPNN ensemble reaches Spearman 0.469 on held aldehyde families and a plain
component-additive ridge reaches 0.469 as well. The measured library is 1,100 lipids. Nothing
in that regime justifies a surrogate whose posterior we cannot write down, and the whole point
of the surrogate here is the posterior: it is what tells us which experiment to run next.

So this is Bayesian linear regression over an explicitly role-decomposed feature map,

    phi(x) = [ a_G phi(G) | a_a phi(m_a) | a_d phi(m_d) | a_i phi(m_i) ],

which gives a closed-form posterior over weights, a closed-form predictive variance, and a
closed-form variance reduction for a candidate batch. The role blocks are the reason the
model is worth writing at all: measuring one lipid informs every other generated design that
shares its head, or its aldehyde tail, or its isocyanide tail, and that structure is exactly
what the reaction-resolved representation hands us for free.

The model is deliberately not a Gaussian process with a learned kernel. Learning kernel
hyperparameters on 1,100 points, then using the resulting posterior to choose experiments,
would put the uncertainty estimate and the acquisition in a feedback loop we cannot audit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ROLES = ("amine", "aldehyde", "isocyanide")


class ReactionFactorizedSurrogateError(RuntimeError):
    """Raised when the surrogate is asked for something its fit does not support."""


@dataclass(frozen=True)
class RoleWeights:
    """Relative weight of each feature block in the kernel.

    product weights whole-molecule similarity; the three role weights control how strongly a
    measurement transfers along each precursor axis. Equal weights are the honest default and
    are what the frozen config uses; anything else is a fitted choice and must be declared.
    """

    product: float = 1.0
    amine: float = 1.0
    aldehyde: float = 1.0
    isocyanide: float = 1.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.product, self.amine, self.aldehyde, self.isocyanide)

    def validate(self) -> None:
        for name, value in zip(("product", *ROLES), self.as_tuple(), strict=True):
            if not np.isfinite(value) or value < 0:
                raise ReactionFactorizedSurrogateError(
                    f"role weight {name!r} must be finite and non-negative, got {value!r}"
                )
        if sum(self.as_tuple()) <= 0:
            raise ReactionFactorizedSurrogateError("at least one role weight must be positive")


def morgan_bits(smiles: str, *, n_bits: int, radius: int) -> np.ndarray:
    """Count-based Morgan features folded to n_bits.

    Counts rather than bits, because binary folded fingerprints are blind to alkyl chain
    length: this project measured 30 generated pairs at Tanimoto 1.000 differing by 14 to 182
    daltons. A surrogate that cannot see a CH2 cannot be trusted to choose between homologues,
    and homologues are exactly what the generator produces.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ReactionFactorizedSurrogateError(f"unparseable SMILES: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    counts = generator.GetCountFingerprintAsNumPy(molecule)
    return np.asarray(counts, dtype=np.float64)


def featurize(
    records: Sequence[dict[str, str]],
    *,
    weights: RoleWeights,
    n_bits: int = 1024,
    radius: int = 2,
    cache: dict[tuple[str, int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Stack the four weighted feature blocks into one design matrix.

    Each record needs keys ``product``, ``amine``, ``aldehyde`` and ``isocyanide`` holding
    canonical SMILES. Blocks are scaled by the square root of their weight so that the induced
    inner product is the weighted sum of per-block inner products.
    """
    weights.validate()
    cache = {} if cache is None else cache
    scales = dict(zip(("product", *ROLES), weights.as_tuple(), strict=True))

    def block(smiles: str) -> np.ndarray:
        key = (smiles, n_bits, radius)
        if key not in cache:
            cache[key] = morgan_bits(smiles, n_bits=n_bits, radius=radius)
        return cache[key]

    rows = []
    for record in records:
        parts = []
        for name in ("product", *ROLES):
            missing = name not in record
            if missing:
                raise ReactionFactorizedSurrogateError(f"record is missing {name!r}: {record}")
            parts.append(np.sqrt(scales[name]) * block(record[name]))
        rows.append(np.concatenate(parts))
    return np.vstack(rows)


@dataclass
class ReactionFactorizedSurrogate:
    """Bayesian linear regression with a role-decomposed feature map.

    Prior: w ~ N(0, prior_variance I). Noise: y = phi(x)^T w + eps, eps ~ N(0, noise_variance).
    Everything downstream reads the posterior covariance, so both are explicit inputs rather
    than fitted quantities; the frozen config sets them and the benchmark holds them fixed
    across every arm so the comparison is about acquisition, not about tuning.
    """

    prior_variance: float = 1.0
    noise_variance: float = 1.0
    weights: RoleWeights = RoleWeights()
    n_bits: int = 1024
    radius: int = 2

    _mean: np.ndarray | None = None
    _covariance: np.ndarray | None = None
    _cache: dict[tuple[str, int, int], np.ndarray] | None = None
    _target_mean: float = 0.0

    def __post_init__(self) -> None:
        if self.prior_variance <= 0 or self.noise_variance <= 0:
            raise ReactionFactorizedSurrogateError("prior and noise variance must be positive")
        self.weights.validate()
        self._cache = {}

    # ------------------------------------------------------------------ fitting

    def design_matrix(self, records: Sequence[dict[str, str]]) -> np.ndarray:
        return featurize(
            records,
            weights=self.weights,
            n_bits=self.n_bits,
            radius=self.radius,
            cache=self._cache,
        )

    def fit(
        self, records: Sequence[dict[str, str]], targets: Sequence[float]
    ) -> ReactionFactorizedSurrogate:
        """Closed-form posterior. Targets are centred; the offset is restored on predict."""
        if len(records) != len(targets):
            raise ReactionFactorizedSurrogateError(
                f"{len(records)} records against {len(targets)} targets"
            )
        if not records:
            raise ReactionFactorizedSurrogateError("cannot fit on an empty set")
        phi = self.design_matrix(records)
        y = np.asarray(targets, dtype=np.float64)
        self._target_mean = float(y.mean())
        centred = y - self._target_mean

        dim = phi.shape[1]
        precision = phi.T @ phi / self.noise_variance + np.eye(dim) / self.prior_variance
        covariance = np.linalg.inv(precision)
        self._covariance = covariance
        self._mean = covariance @ (phi.T @ centred) / self.noise_variance
        return self

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray]:
        if self._mean is None or self._covariance is None:
            raise ReactionFactorizedSurrogateError("surrogate has not been fitted")
        return self._mean, self._covariance

    # ---------------------------------------------------------------- prediction

    def predict(self, records: Sequence[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior predictive mean and standard deviation, including noise."""
        mean, covariance = self._require_fit()
        phi = self.design_matrix(records)
        mu = phi @ mean + self._target_mean
        var = np.einsum("ij,jk,ik->i", phi, covariance, phi) + self.noise_variance
        return mu, np.sqrt(np.maximum(var, 0.0))

    def lower_bound(self, records: Sequence[dict[str, str]], *, z: float = 1.2816) -> np.ndarray:
        """Conservative score. Default z is the one-sided 90% normal quantile.

        This is a Bayesian credible bound under the model, not a conformal guarantee. The
        conformal machinery elsewhere in the project makes a distribution-free statement; this
        one is only as good as the linear-Gaussian assumption, and is labelled accordingly.
        """
        mu, sd = self.predict(records)
        return mu - z * sd

    # ------------------------------------------------------- experimental design

    def variance_over(self, records: Sequence[dict[str, str]]) -> float:
        """Total posterior variance of the latent function over a target population."""
        _, covariance = self._require_fit()
        phi = self.design_matrix(records)
        return float(np.einsum("ij,jk,ik->i", phi, covariance, phi).sum())

    def covariance_after(self, batch: Sequence[dict[str, str]]) -> np.ndarray:
        """Posterior covariance that would follow from observing ``batch``.

        Uses the Woodbury update, so the cost is cubic in the batch size rather than in the
        feature dimension. This never touches the observed values: the covariance of a
        Gaussian posterior does not depend on the labels, which is what makes the design
        problem solvable before the experiment is run.
        """
        _, covariance = self._require_fit()
        if not batch:
            return covariance
        phi = self.design_matrix(batch)
        middle = phi @ covariance @ phi.T + self.noise_variance * np.eye(len(batch))
        return covariance - covariance @ phi.T @ np.linalg.solve(middle, phi @ covariance)

    def variance_reduction(
        self,
        batch: Sequence[dict[str, str]],
        target: Sequence[dict[str, str]],
    ) -> float:
        """Expected reduction in total posterior variance over ``target`` from ``batch``."""
        _, covariance = self._require_fit()
        phi_target = self.design_matrix(target)
        before = float(np.einsum("ij,jk,ik->i", phi_target, covariance, phi_target).sum())
        after_cov = self.covariance_after(batch)
        after = float(np.einsum("ij,jk,ik->i", phi_target, after_cov, phi_target).sum())
        return before - after

    # The greedy batch loop below needs the variance reduction of every candidate against a
    # posterior that changes at each step. Doing that with covariance_after per candidate is
    # correct but wasteful. Observing one point is a rank-one update, so the reduction in total
    # variance over a target population T has a closed form. Writing M = Phi_T^T Phi_T,
    #
    #     V(Sigma) = tr(Sigma M),
    #     Sigma'   = Sigma - Sigma phi phi^T Sigma / (phi^T Sigma phi + sigma^2),
    #     dV(phi)  = phi^T (Sigma M Sigma) phi / (phi^T Sigma phi + sigma^2).
    #
    # Sigma M Sigma is formed once per greedy step and every candidate is then two quadratic
    # forms, which is what makes a pool of thousands tractable.

    def target_gram(self, target: Sequence[dict[str, str]]) -> np.ndarray:
        """M = Phi_T^T Phi_T for a target population, computed once per round."""
        phi = self.design_matrix(target)
        return phi.T @ phi

    def variance_reduction_scores(
        self,
        candidates: Sequence[dict[str, str]],
        gram: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """Closed-form variance reduction over the target for each candidate, vectorised."""
        phi = self.design_matrix(candidates)
        middle = covariance @ gram @ covariance
        numerator = np.einsum("ij,jk,ik->i", phi, middle, phi)
        denominator = np.einsum("ij,jk,ik->i", phi, covariance, phi) + self.noise_variance
        return np.maximum(numerator / denominator, 0.0)

    def rank_one_update(self, covariance: np.ndarray, record: dict[str, str]) -> np.ndarray:
        """Posterior covariance after observing one further design."""
        phi = self.design_matrix([record])[0]
        scaled = covariance @ phi
        return covariance - np.outer(scaled, scaled) / (float(phi @ scaled) + self.noise_variance)

    def variance_over_gram(self, gram: np.ndarray, covariance: np.ndarray) -> float:
        """tr(Sigma M): total posterior variance of the latent function over the target."""
        return float(np.sum(covariance * gram.T))

    def role_variance_over(self, records: Sequence[dict[str, str]]) -> dict[str, float]:
        """Total posterior variance attributable to each feature block.

        Diagnostic rather than decisional. It answers which precursor axis the surrogate is
        currently most ignorant about, which is the question a chemist asks when deciding what
        to vary next.
        """
        _, covariance = self._require_fit()
        phi = self.design_matrix(records)
        width = self.n_bits
        out: dict[str, float] = {}
        for index, name in enumerate(("product", *ROLES)):
            lo, hi = index * width, (index + 1) * width
            block = phi[:, lo:hi]
            sub = covariance[lo:hi, lo:hi]
            out[name] = float(np.einsum("ij,jk,ik->i", block, sub, block).sum())
        return out

    def state(self) -> dict[str, Any]:
        mean, covariance = self._require_fit()
        return {
            "feature_dimension": int(mean.size),
            "prior_variance": self.prior_variance,
            "noise_variance": self.noise_variance,
            "role_weights": dict(zip(("product", *ROLES), self.weights.as_tuple(), strict=True)),
            "n_bits": self.n_bits,
            "radius": self.radius,
            "target_mean": self._target_mean,
            "posterior_trace": float(np.trace(covariance)),
        }
