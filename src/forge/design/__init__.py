"""Designing whole lipids: the generative model and everything that feeds or follows it.

    flow/       the model's architecture and graph representation
    training/   fitting it
    sampling/   drawing molecules from it
    corpus/     what it is fitted and evaluated on
    guidance/   which molecules to favour, and how strongly
    schedule/   orchestrating a production run
    audit/      describing what came out

Called `product` until the split above made the scope plain. It does not only produce reaction
products -- it trains the model, curates corpora, schedules runs and decides eligibility -- and
"product" reads as a build artifact to anyone who has not met the chemistry sense of the word.
`generate` was the other candidate and is too narrow: generation is one of these seven.

The subpackages stay under one parent rather than being promoted to `forge.flow`,
`forge.corpus` and so on. `audit` and `corpus` already exist at the top level, and `audit` and
`guidance` are subpackages of `route`, `potency` and `value` too -- auditing a retrosynthesis
engine is not auditing a generated molecule. The parent is what lets those names mean different
things in different places, so an import line says which one it meant.

This package re-exports nothing. A consumer names the subpackage it depends on, and that
dependency is visible in the import.
"""
