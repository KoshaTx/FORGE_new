# Multireaction experiment application

This application supports the original three-family study and the newer 22-family
COMPOSE study. Keep both studies and their recorded results reproducible. The two
studies share reusable code in `forge/` and some experiment infrastructure here;
a module's age alone does not make it dead code.

- `specifications.py` owns this application's CLI experiment IDs and JSON paths.
- `foundation_stages.py` contains the original multireaction corpus, training,
  sampling, and overfit stage adapters.
- `reaction_specialization_stages.py` contains specialist training and preflight
  stage adapters.
- `compose_lipid_training.py` contains the 22-family COMPOSE training adapter.
- `stages.py` contains the remaining shared adapters and re-exports moved stage
  functions so existing imports still work.

The catalog in `experiments/catalog.py` loads the registered stages. The JSON
specifications and frozen configs retain their existing identifiers and paths.
