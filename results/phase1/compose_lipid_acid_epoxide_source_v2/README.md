# Xu acid/epoxide source qualification — 2026-09-22

**All 101,098 currently eligible acid/epoxide records reconstruct exactly.** The complete protected
source partition was used, with no record or molecular-size cap. The 395 restartable CPU shards took
435.17 seconds. Independent reconciliation checked every target, complete global component ID,
role, quantity, unique forward/inverse result and inventory balance against the source joins.

This is computed transform consistency, not a claim that those generated compounds were made.
Coverage is 101,098/101,098 eligible records, and exactness among attempted reconstructions is
101,098/101,098. Neither ratio estimates experimental success. The family reaches 56 heavy atoms
in this eligible population; no size filter was applied. Protected rows remain excluded.

## Primary evidence and discrepancy

The supplied `adhm202302691-sup-0001-suppmat.pdf` is the SI for DOI 10.1002/adhm.202302691.
PDF pp. 2–4 and 20–22 were visually reviewed. Fig. S1 on p. 2 specifies two ordered stages:

1. E12 terminal epoxide and CA1 hydrophobic acid produce the primary ester with the secondary OH
   retained: FeCl3/pyridine, room temperature, 12 h.
2. The remaining OH is esterified with the complete amine-bearing acid A3: EDC hydrochloride,
   DMAP and DIPEA, dichloromethane, room temperature, 12 h.

The p. 3 E12CA1A3 drawing contains the 4-(dimethylamino)butanoyl head, while its printed name says
3-(dimethylamino)propanoyl. The drawing gives neutral C34H67NO4 and calculated protonated mass
554.514286, agreeing with the reported C34H68NO4+ calculated/found values 554.5143/554.5173.
The shorter printed name gives one fewer carbon. Fig. S23 step f on p. 21 explicitly names the
four-carbon acid, and the p. 22 analogue drawing/name corroborates that head identity.

The conflict is retained in `adjudication.json`. The independent positive control uses the drawn
structure. The shorter named product and the same-formula product with head/hydrophobe positions
exchanged are rejected for those exact source inputs. This does not exclude shorter amino-acid
homologues supplied under their own complete component identities. No exact-execution, yield,
purity, biological or training label is transferred. Fig. S1 does not supply complete quantities,
workup or isolated yield. The C3/C4/C5 analogues use a separate diol route and do not expand the
epoxide-opening claim. Oxygen atom mapping is a declared transform convention, not isotope evidence.

The additive FORGE registry derives its terminal-epoxide and carboxylic-acid interfaces from pinned
registries, and derives the new net connectivity from Fig. S1. It retains the existing esterification
interface and its competing-handle guards. Neutral aliphatic CHO hydrophobes and an ionizable CHNO
acid head must preserve all declared roles. All eligible attachment sites and complete inverse
tuples remain visible. Existing registries and upstream COMPOSE were not changed.

## Receipts

- `adjudication.json`, `replay-config.json`: source locators, discrepancy, positive/negative controls,
  parent registry pins and search bounds.
- `control-replays.json`: source-positive reconstruction and both rejected equivalence claims.
- `replay/request.json`, `replay/preflight.json`: exact inputs/code/runtime and serial/parallel identity.
- `replay/result.json`: all 395 completed, pinned shards and the complete family census.
- `reconciled/combined-replay-audit.json`: independent source/partition reconciliation.
- `reconciled/result.json`, `reconciled/evidence.sqlite`: previous evidence retained and only the new
  family evidence appended; no rebuild of the existing chemistry or tensor caches.
- `targeted-tests.xml`: **11 passed in 1.694 seconds**, with no failures/errors/skips. The checks cover
  the independent intermediate/product, formula/mass discrepancy, wrong site order, competing
  handles, global quantities, atom-order invariance and incomplete-search rejection.
- `vendor-verify.log`: all 30 manifest assets verified. New registry/source pins are additionally
  verified by the source loader, replay and reconciliation. No full-suite rerun was performed.

`initial-origin-control.json` is a diagnostic successful source-control atom trace, not a completed
model-input dataset. Population-wide atom origins and tensor preparation remain to be done for
the new AEMA and acid/epoxide evidence.

Recheck this checkpoint:

```sh
PYTHONPATH=. .venv/bin/python results/phase1/compose_lipid_acid_epoxide_source_v2/finalize.py --verify
```

The replay resumes completed shards only when source/configuration/implementation hashes match;
completed final outputs are not overwritten. No training or paid compute ran.

The current readiness checkpoint is `../compose_lipid_training_readiness_v13/readiness.json`:
**1,024,230 exact records across 19 families**, **248,236 eligible records still pending chemistry**.
Model-input preparation still covers 919,567 records; the additional 104,663 await preparation.
