"""Potency prediction: how well a lipid is likely to work, and whether we may say so.

    oracle/         fitting and freezing the predictors
    applicability/  whether a candidate is inside the calibrated domain
    morphology/     morphology-conditioned proposals and challengers
    ranking/        ordering once potency and applicability are known
    audit/          post-hoc description of the above

`applicability` gates `ranking`: a prediction outside the calibrated domain abstains rather
than scoring low, which is the rule that keeps an extrapolation from reaching guidance.
"""
