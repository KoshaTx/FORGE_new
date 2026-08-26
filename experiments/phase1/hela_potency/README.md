# HeLa potency diagnostic

This is the single bounded potency-only pilot authorized by `AGENTS.md`. It owns its applicability
policy, conservative utility, matched in-trajectory/post-hoc arms, and lambda-zero identity check.
It cannot call synthesis/proposal engines, access sealed holdouts, lock candidates, or support an
in-vivo efficacy claim.

`data.json` is the supported biological-data entry point. LNPDB is its only row-level input for
AGILE, JC and LM. The resulting observation table excludes every AGILE B4 mixture measurement,
canonicalizes complete lipid graphs under the constitution-only Phase 1 identity, keeps every
study/endpoint scale separate, and contains no reaction-program fields. It is the only data shape
new potency experiments should consume.

Build, verify, and reproduce it with:

```bash
make phase1-potency-study-corpus
```

Library code imports `load_potency_study_corpus` and `StudyEndpoint` from
`forge.potency.study_data`, then must request a study/endpoint pair; there is no cross-study
target-array API. Exact reaction-program supervision is loaded independently from its own derived
corpus. Existing M0 and completed diagnostic artifacts remain frozen and are not rewritten through
this view.
