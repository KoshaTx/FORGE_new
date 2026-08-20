"""Whether a candidate falls inside the domain the oracle was calibrated on.

Outside it the answer is abstention, not a low score -- an extrapolative prediction must never
reach guidance.
"""
