"""Public Phase 1 corpus contract.

The implementation remains byte-compatible with the frozen ``forge.design.corpus.phase1_data`` module
during migration.  New callers use this domain API; the legacy import stays available for frozen
scripts and result provenance.
"""

from forge.design.corpus.phase1_data import Phase1DataError, freeze_phase1_data_contract

__all__ = ["Phase1DataError", "freeze_phase1_data_contract"]
