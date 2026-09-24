"""Only new receipt-pinned positives enter the incremental origin population."""

from results.phase1.compose_lipid_new_supplement_origins_v1.run import selected_records


def test_receipt_objects_select_new_positives_without_reencoding_existing_rows():
    items = [
        {"evidence": {"receipt": {"path": "old.json", "sha256": "old"}}},
        {"evidence": {"receipt": {"path": "new.json", "sha256": "new"}}},
    ]

    class Reader:
        def iter_preparation_records(self, *, family, exact):
            assert family == "amine_alkylation" and exact is True
            yield from items

    found = list(selected_records(Reader(), "amine_alkylation", {"new_evidence_receipts": ["new"]}))
    assert found == items[1:]
    assert found[0] is items[1]
