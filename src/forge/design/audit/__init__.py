"""Post-hoc audits over generated products.

An audit describes a result; it must never select on one. Kept beside the code it audits rather
than in the top-level `forge.audit`, because these are imported by their siblings -- moving them
out would build a hub rather than a boundary.
"""
