# M0-07 production-oracle boundary

The production result is the inference trust root for the frozen AGILE oracle.
Inference must fail closed unless it authenticates all of the following:

- the production configuration and its declared result path;
- the scientific freeze result and selected model;
- the selected representation-lane result and source configuration;
- the checkpoint path, byte count, format, and SHA-256 hash;
- the complete data policy and software-version contract;
- the measured-component applicability index;
- the qualified Ugi reaction registry and reaction variant.

The component index is not trusted merely because it is serialized inside the
checkpoint. The loader rebuilds it from the hash-pinned 1,100-record curated
dataset, including canonical component identities and measured component
triples, and requires exact equality with the checkpoint. The loader likewise
rebuilds the Ugi chemistry contract from its pinned registry and variant and
requires exact equality.

Candidate applicability is derived rather than caller-declared. A candidate
must forward-reconstruct its stated product through the checkpoint-owned
AGILE-type Ugi 3-CR transform. Canonical amine, aldehyde, and isocyanide
identities are then compared with the checkpoint-owned measured component
sets. This determines the component-novelty domain. Unsupported domains,
including three unseen components, non-Ugi final assembly, and in-vivo use,
retain explicit abstention.

Production graph epoch counts may use only fit files listed by the selected
lane's frozen fit-source manifest. Every accepted fit must preserve:

- its fit schema, completion status, job ID, architecture, endpoint, and seed;
- its exact fold and repository-relative output path;
- the selected-lane configuration hash and complete input-hash map;
- the train-only epoch-selection boundary;
- an epoch within the frozen maximum;
- the boundary proving that calibration and test targets affected neither
  scaling, epoch selection, nor weight updates.

These checks harden artifact integrity and applicability enforcement. They do
not alter the selected architecture, reinterpret outer-test metrics, or change
the universal guidance-abstention decision.
