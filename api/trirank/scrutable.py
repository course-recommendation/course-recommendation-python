"""TriRank variant supporting scrutable, per-request aspect preference overrides
without mutating the shared graph structure (R, X, Y).

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

`ScrutableTriRank` keeps `self.Y` untouched and instead lets callers supply
`a0_overrides`: an `{aspect_idx: weight}` mapping with weights already in
[0, 1].

Blending rather than overwriting
--------------------------------
An earlier version of this class wrote the overrides into the already
L1-normalized `a_0` and then renormalized. That has two problems:

1. It *replaces* whatever the user's reviews said about an overridden
   aspect instead of combining the two, so an aspect the caller happens to
   send with weight 0 silently erases the evidence from `self.Y`.
2. L1 normalization cancels any uniform scaling, so a preference vector
   that is flat - every aspect requested equally - renormalizes to exactly
   the same thing no matter what level it was sent at. Measured on the real
   dataset, "every aspect at the minimum" and "every aspect at the maximum"
   produced byte-identical `a_0` and an identical ranking for all 179
   items.

So the override vector is now normalized on its own and mixed with the
review-derived prior at `a0_prior_weight`, which keeps both signals present
and makes the mixing ratio an explicit, tunable quantity rather than an
accident of normalization.

Note that (2) is only fully fixed in combination with a bipolar aspect
vocabulary on the caller's side: the two poles of an axis have to be
*distinct aspects* for "I want the low end everywhere" and "I want the high
end everywhere" to be different vectors at all.
"""

import numpy as np
from cornac.models import TriRank

EPS = 1e-10

#: Default weight of the review-derived prior when blending in explicit
#: overrides. 0.3 leaves stated preferences clearly dominant while keeping the
#: user's reviews in play. There is no usage data to tune this against yet.
DEFAULT_A0_PRIOR_WEIGHT = 0.3


class ScrutableTriRank(TriRank):
    """TriRank variant that lets callers inject an explicit aspect
    preference for a single online recommendation call, without mutating the
    shared Y matrix used for graph smoothing.

    Attributes
    ----------
    a0_overrides: dict[int, float]
        Mapping of aspect index -> preference weight in [0, 1], blended on
        top of the aspect preference vector (a_0) derived from the user's
        reviews before it is used in the iterative ranking update.
    a0_prior_weight: float
        Weight in [0, 1] given to the review-derived prior when blending.
        0 ignores the user's reviews entirely; 1 ignores the overrides.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a0_overrides: dict[int, float] = {}
        self.a0_prior_weight: float = DEFAULT_A0_PRIOR_WEIGHT

    def _blend_aspect_preference(self, a_0):
        """Mix the caller's aspect preferences into the L1-normalized `a_0`."""
        overrides = np.zeros_like(a_0)
        for aspect_idx, weight in self.a0_overrides.items():
            overrides[aspect_idx] = min(max(float(weight), 0.0), 1.0)
        if not overrides.any():
            return a_0

        overrides /= np.linalg.norm(overrides, 1)
        prior_weight = min(max(float(self.a0_prior_weight), 0.0), 1.0)
        blended = prior_weight * a_0 + (1.0 - prior_weight) * overrides
        if blended.any():
            blended /= np.linalg.norm(blended, 1)
        return blended

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

        if self.a0_overrides:
            a_0 = self._blend_aspect_preference(a_0)

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
                # eq. 4 weights the aspect fitting term by eta_A, not eta_P.
                # Upstream cornac uses eta_P here; with every weight at the
                # default of 1 the two coincide, which is why it goes
                # unnoticed, but it makes eta_A - the one knob for how hard
                # the stated aspect preference is enforced - a no-op.
                + self.eta_A / a_denominator * a_0
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
        model.a0_prior_weight = DEFAULT_A0_PRIOR_WEIGHT
    return model
