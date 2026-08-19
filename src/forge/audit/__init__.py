"""Post-hoc audits: analyses that describe a result without being allowed to select on it.

The distinction this package exists to make visible is scientific, not organisational. An audit
answers "is this claim supported?" after the fact; it must never feed the decision it is auditing.
`EvidenceStatus.VERIFIED_NONSELECTING` and `DIAGNOSTIC_ONLY` encode the same rule in the evidence
vocabulary, and `ugi_distribution_overlap`'s own docstring states it directly -- the audit "does
not estimate a probability-density ratio or authorize biological guidance."

Collected from `eval/` and `verify/`, which were two activity-named folders holding one audit each
while the rest of the tree is named by domain. Both are retired.

**This is a partial migration, deliberately.** 56 modules in this package look like audits, but 23
of them are imported by other modules, carrying 179 inbound edges between them. Moving those here
would not create a boundary -- it would create a hub that 106 modules depend on, which is the same
mistake the `bio` -> `potency` move made when it relocated edges instead of removing them. Only
audits that nothing imports belong here. These two qualify: neither had a single importer inside
`src/forge`.
"""

from forge.audit.flower_transfer_audit import (
    FlowerTransferAuditError,
    audit_flower_transfer_inputs,
    run_flower_transfer_audit,
)
from forge.audit.ugi_distribution_overlap import build_ugi_distribution_overlap_audit

__all__ = [
    "FlowerTransferAuditError",
    "audit_flower_transfer_inputs",
    "build_ugi_distribution_overlap_audit",
    "run_flower_transfer_audit",
]
