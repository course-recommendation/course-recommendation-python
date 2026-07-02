"""TriRank variant that supports scrutable, per-request aspect preference
overrides without mutating the shared graph structure (R, X, Y).

Background
----------
`cornac.models.TriRank._online_recommendation` builds the aspect prior
preference vector `a_0` directly from a row of `self.Y` (the symmetrically
normalized user-aspect matrix). `self.Y` is also used as the smoothness
structure (`S_Y` in the paper) inside the iterative update rules (`Y * a`,
`Y.T * u`), shared across every user scored by the model.

Writing a raw, arbitrary-scale user preference (e.g. a 1-5 rating) directly
into `self.Y[user_idx, aspect_idx]` is incorrect for two reasons:

1. Every legitimately-computed entry of `self.Y` is mathematically bounded
   in [0, 1] (a consequence of the symmetric normalization), and in
   practice entries are usually much smaller than 1. A raw score of 1-5 is
   out of range and, once `a_0` is L1-normalized in
   `_online_recommendation`, will dominate the entire aspect preference
   vector, drowning out the aspects actually inferred from the user's
   reviews.
2. It corrupts `self.Y`, which is also used as the shared smoothness
   matrix for every user/aspect computed during the update loop -
   conflating the "fitting prior" role (Algorithm 1, step 5) with the
   "graph edge weight" role (Algorithm 1, step 4) that the paper keeps
   separate.

`ScrutableTriRank` fixes this by keeping `self.Y` untouched and instead
letting callers supply `a0_overrides`: a `{aspect_idx: weight}` mapping
where `weight` is already normalized to [0, 1]. The override is merged
into `a_0` *after* it has been L1-normalized from `self.Y[user]`, and the
vector is renormalized afterwards so it remains a valid preference
distribution.
"""

import numpy as np
from cornac.models import TriRank

EPS = 1e-10


class ScrutableTriRank(TriRank):
    """TriRank variant that lets callers inject an explicit aspect
    preference override for a single online recommendation call, without
    mutating the shared Y matrix used for graph smoothing.

    Attributes
    ----------
    a0_overrides: dict[int, float]
        Mapping of aspect index -> preference weight in [0, 1]. Applied on
        top of the aspect preference vector (a_0) derived from the user's
        reviews before it is used in the iterative ranking update.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a0_overrides: dict[int, float] = {}

    def _online_recommendation(self, user):
        # Algorithm 1: Online recommendation line 5
        p_0 = self.r_mat[[user]]
        p_0.data.fill(1)
        p_0 = p_0.toarray().squeeze()
        a_0 = self.Y[user].toarray().squeeze().astype(float)
        u_0 = np.zeros(self.r_mat.shape[0])
        u_0[user] = 1

        # Algorithm 1: Online recommendation line 6
        if p_0.any():
            p_0 /= np.linalg.norm(p_0, 1)
        if a_0.any():
            a_0 /= np.linalg.norm(a_0, 1)
        if u_0.any():
            u_0 /= np.linalg.norm(u_0, 1)

        # Merge in explicit user overrides (already normalized to [0, 1])
        # on the same scale as the L1-normalized a_0, then renormalize so
        # the result remains a valid L1 preference vector.
        if self.a0_overrides:
            for aspect_idx, weight in self.a0_overrides.items():
                a_0[aspect_idx] = weight
            if a_0.any():
                a_0 /= np.linalg.norm(a_0, 1)

        # Algorithm 1: Online recommendation line 7
        p = self.p.copy()
        a = self.a.copy()
        u = self.u.copy()

        # Algorithm 1: Online recommendation line 8
        prev_p = p
        prev_a = a
        prev_u = u
        inc = 1
        while True:
            # eq. 4
            u_denominator = self.alpha + self.gamma + self.eta_U + EPS
            u = (
                self.alpha / u_denominator * self.R * p
                + self.gamma / u_denominator * self.Y * a
                + self.eta_U / u_denominator * u_0
            ).squeeze()
            p_denominator = self.alpha + self.beta + self.eta_P + EPS
            p = (
                self.alpha / p_denominator * self.R.T * u
                + self.beta / p_denominator * self.X * a
                + self.eta_P / p_denominator * p_0
            ).squeeze()
            a_denominator = self.gamma + self.beta + self.eta_A + EPS
            a = (
                self.gamma / a_denominator * self.Y.T * u
                + self.beta / a_denominator * self.X.T * p
                + self.eta_P / a_denominator * a_0
            ).squeeze()

            if (self.max_iter > 0 and inc > self.max_iter) or (
                np.all(np.isclose(u, prev_u))
                and np.all(np.isclose(p, prev_p))
                and np.all(np.isclose(a, prev_a))
            ):  # stop when converged
                break
            prev_p, prev_a, prev_u = p, a, u
            inc += 1

        # Algorithm 1: Online recommendation line 9
        return p, a, u


def to_scrutable(model: TriRank) -> ScrutableTriRank:
    """Reclass an already-trained/loaded TriRank instance in place so it
    gains the `a0_overrides` mechanism, without needing to retrain."""
    if not isinstance(model, ScrutableTriRank):
        model.__class__ = ScrutableTriRank
        model.a0_overrides = {}
    return model
