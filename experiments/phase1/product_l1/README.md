# Phase 1 product + L1

This application owns one scientific model and two explicit pipelines.

```bash
forge doctor phase1-training-smoke
forge experiment run phase1-training-smoke --profile smoke --resume

forge doctor phase1-sampling
forge experiment run phase1-sampling --profile smoke --resume
forge experiment reproduce phase1-sampling --profile smoke
```

`training/` implements cache preparation and resumable joint/closure training. `sampling/`
implements restartable, sharded terminal generation and exact Ugi-L1 admission. `stages.py` is the
thin orchestration adapter; it must not define chemistry or model policy.

The production training specification uses the same APIs with an explicit L4 resource envelope and
is never launched by smoke tests. Sampling does not retrain, retry/repair, call a route planner, call
a potency oracle, or lock candidates.
