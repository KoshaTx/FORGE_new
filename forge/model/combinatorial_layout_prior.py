"""Count-only layout sampling for every admitted combinatorial program.

Multiple source topologies at a depth do not invalidate a count-only distribution. This prior
does not assert a unique exact topology; every generated product still needs full L1 checking.
"""

from collections import Counter, defaultdict

from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutError,
    SynthesisProgramLayoutPrior,
    _compile_program_distribution,
    _summarize_program,
)


class CombinatorialLayoutPrior(SynthesisProgramLayoutPrior):
    """Reuse the frozen count law without requiring unique nonfixed core topologies."""

    def __init__(self, cache):
        self.vocabulary = cache.vocabulary
        self.maximum_heavy_atoms = int(cache.metadata["support"]["maximum_heavy_atoms"])
        self.maximum_closures = int(cache.metadata["support"]["maximum_closures"])
        self._distributions = {}
        self._fixed_node_states = {}
        self._program_topology_signatures = {}
        self._repeated_role_states = {}
        self._conditioned_component_sizes = {}
        self._ugi_measured_joint_layout_support = None
        self.topology_support_counts = {}
        for family in cache.vocabulary.program_states[1:]:
            summaries, fixed = _summarize_program(cache, family)
            self._distributions[family] = _compile_program_distribution(summaries)
            self._fixed_node_states[family] = fixed
            topologies, multiplicity = defaultdict(set), Counter()
            for summary in summaries:
                topologies[summary.depth].add(summary.program_topology_signature)
                for role, count in Counter(role for role, _, _ in summary.blocks).items():
                    multiplicity[role] = max(multiplicity[role], count)
            self.topology_support_counts[family] = {
                depth: len(values) for depth, values in topologies.items()
            }
            repeated = [role for role, count in multiplicity.items() if count > 1]
            if len(repeated) > 1:
                raise SynthesisProgramLayoutError(
                    "multiple repeated roles require a richer count prior: " + family
                )
            self._repeated_role_states[family] = repeated[0] if repeated else None
            # No nonfixed topology is promoted to a unique program. This path supports only
            # count draws; the unchanged adapter-fixed chemistry is still retained.
            self._program_topology_signatures[family] = {}
        self.support_report = self.validate_support()

    def sample(self, program_id, *, sample_count, seed):
        return super().sample(
            program_id,
            sample_count=sample_count,
            seed=seed,
            role_morphology_conditioning=False,
            exact_program_topology=False,
        )
