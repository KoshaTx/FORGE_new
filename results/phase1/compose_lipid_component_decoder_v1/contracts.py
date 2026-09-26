"""Complete chemical contracts without opening preparation or holdout corpora."""

import hashlib
import json
from functools import partial

from rdkit import Chem

from forge.assembly.condensation_event import check_condensation_event
from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.generated_source import assess_product
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.assembly.sequential_program import RegistrySequentialProgram
from forge.assembly.source_event import check_source_event
from forge.core.hashing import resolve_pin
from forge.corpus import compose_lipid_current_replay as current
from forge.corpus import compose_lipid_fixed_profiles as profiles
from forge.corpus import compose_lipid_fixed_replay as fixed
from forge.corpus import compose_lipid_grouped_replay as grouped
from forge.corpus import compose_lipid_source_program as passerini
from forge.corpus import compose_lipid_staged_replay as staged
from forge.corpus.compose_lipid_family_replay import _passerini_replay
from forge.corpus.compose_lipid_source_view import pin
from results.phase1.compose_lipid_family_rules_v2.run import MIRROR, ROOT
from results.phase1.compose_lipid_family_rules_v2.score import target_executors


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def read(path):
    return json.loads(path.read_text())


def adapter_from(cfg, reaction_id):
    p = cfg["inputs"]["registry"]
    return RegistryAssemblyAdapter.from_registry(
        resolve_pin(p, MIRROR, label="registry"),
        reaction_id=reaction_id,
        expected_sha256=p["sha256"],
    )


def full_fixed(executor, *, bound=256):
    run = executor["run"]
    base = run
    while not isinstance(base.args[0], RegistryAssemblyAdapter):
        base = base.args[0]
    adapter = base.args[0]
    return {
        **executor,
        "adapter": adapter,
        "bounds": RepeatBounds(maximum_events=1, maximum_outcomes=bound),
        "kind": "fixed",
        "full_source_contract": True,
        "reaction": next(
            r
            for r in read(adapter.registry_path)["reactions"]
            if r["reaction_id"] == adapter.reaction_id
        ),
    }


def single_event(path):
    """Same adjudicated reactive-site kwargs as preparation, without its reader."""
    cfg = read(path)
    paths = {k: resolve_pin(v, MIRROR, label=k) for k, v in cfg["inputs"].items()}
    source = read(paths["adjudication"])
    assert source["family"] == cfg["family"] and source["reaction_id"] == cfg["reaction_id"]
    for name in ("registry", "functional_group_registry"):
        assert source[name] == cfg["inputs"][name]
    for k, v in source["assets"].items():
        resolve_pin(v, MIRROR, label=k)
    contract = source["source_contract"]
    assert contract["roles"] == read(paths["decomposition_key"])[cfg["family"]]["roles"]
    adapter = adapter_from(cfg, cfg["reaction_id"])
    queries = {}
    for role, ref in contract["query_references"].items():
        reaction = next(
            r
            for r in read(paths[ref["input"]])["reactions"]
            if r["reaction_id"] == ref["reaction_id"]
        )
        queries[role] = Chem.MolFromSmarts(
            next(r for r in reaction["reactant_roles"] if r["name"] == ref["role"])[
                "required_handle_smarts"
            ]
        )
    kwargs = dict(
        site_contract=contract["site_contract"],
        role_queries=queries,
        maximum_outcomes=cfg["maximum_outcomes"],
    )
    reaction = next(
        r for r in read(paths["registry"])["reactions"] if r["reaction_id"] == adapter.reaction_id
    )
    function = check_source_event
    if "net_byproducts" in contract:
        assert contract["net_byproducts"] == reaction["net_byproducts"]
        kwargs["net_byproducts"] = reaction["net_byproducts"]
        function = check_condensation_event
    run = partial(function, adapter, **kwargs)
    controls = {}
    for c in source["source_controls"]:
        checked = run(c["components"], c["expected_product"])
        assert checked["computed_consistency_pass"], c["label"]
        controls[c["label"]] = checked
    return (
        cfg,
        full_fixed(
            dict(run=run, mapping=contract["registry_to_source_roles"], source_contract=contract),
            bound=cfg["maximum_outcomes"],
        ),
        controls,
    )


def load_all():
    found, inputs, controls = {}, {}, {}

    def cfg_path(relative):
        path = MIRROR / relative
        inputs[relative] = pin(ROOT, path)
        return path

    def put(key, executor):
        e = dict(executor, full_source_contract=True)
        if "mapping" not in e:
            e["mapping"] = e.get("source_roles", {r: r for r in e["program"].roles})
        found.setdefault(key, []).append(e)

    origin = read(cfg_path("configs/multireaction/compose_lipid_program_origins_v1.json"))
    for name, value in origin["contracts"].items():
        path = resolve_pin(value, MIRROR, label=name)
        cfg = read(path)
        if name != "michael":
            cfg, _, adapters, programs, bounds, checked = current.load_contract(MIRROR, path)
        else:
            paths = {k: resolve_pin(v, MIRROR, label=k) for k, v in cfg["inputs"].items()}
            registry = read(paths["registry"])
            for k, v in registry["source_assets"].items():
                resolve_pin(v, MIRROR, label=k)
            adapters, programs = {}, {}
            for family, binding in cfg["families"].items():
                reaction = next(
                    r for r in registry["reactions"] if r["reaction_id"] == binding["reaction_id"]
                )
                adapters[family] = adapter_from(cfg, binding["reaction_id"])
                programs[family] = reaction["source_program"]
            bounds = RepeatBounds(**cfg["search_bounds"])
            doc = read(paths["transform_controls"])
            assert doc["registry"] == cfg["inputs"]["registry"]
            checked = {}
            for kind in ("source_controls", "ambiguity_controls"):
                for c in doc[kind]:
                    family = c["family"]
                    r = replay_repeated_components(
                        adapters[family],
                        c["components"],
                        c["expected_product"],
                        accumulator_role=programs[family]["accumulator_role"],
                        events=c["events"],
                        byproducts_per_event=programs[family]["net_byproducts_per_event"],
                        bounds=bounds,
                    )
                    assert r["computed_consistency_pass"] == (kind == "source_controls")
                    assert r["checks"]["complete_search"]
                    checked[c["label"]] = r
        controls[name] = checked
        for family, adapter in adapters.items():
            if family not in origin["families"]:
                continue
            key = family
            if family == "maleate_addition":
                key += ":thiol" if name == "a3_thiol" else ":amine"
            put(
                key,
                dict(
                    kind="repeated",
                    adapter=adapter,
                    mapping=cfg["families"][family]["registry_to_source_roles"],
                    program=programs[family],
                    bounds=bounds,
                ),
            )
    # Full fixed-event source contracts, including retained heads and profiles.
    _, local, local_inputs, checked = target_executors()
    inputs.update(local_inputs)
    controls["scaffold_and_ketone"] = checked
    for family, values in local.items():
        for e in values:
            if "mapping" not in e:
                e = {**e, "mapping": {r: r for r in e["run"].args[0].roles}}
            put(family, full_fixed(e))
    profile_path = cfg_path(
        "configs/multireaction/compose_lipid_supplied_ketone_ugi4_profiles_v1.json"
    )
    _, _, profiled, checked = profiles.load_contract(MIRROR, profile_path)
    controls["ketone_profiles"] = checked
    for family, e in profiled.items():
        for p in e["source_role_profiles"]:
            put(family, full_fixed({**e, "mapping": p["registry_to_source_roles"]}))
    for label, path in [
        ("ugi3", "configs/multireaction/compose_lipid_v8_ugi3_program_v1.json"),
        ("ugi4", "configs/multireaction/compose_lipid_v8_ugi4_program_v1.json"),
    ]:
        cfg, e, checked = single_event(cfg_path(path))
        controls[label] = checked
        put(cfg["family"], e)
        if label == "ugi3":
            remaining = read(
                cfg_path("results/phase1/compose_lipid_remaining_origins_v2/config.json")
            )
            binding = read(
                resolve_pin(remaining["role_binding"], MIRROR, label="Ugi3 role binding")
            )
            alternative = read(
                resolve_pin(binding["artifacts"]["binding.json"], MIRROR, label="Ugi3 namespaces")
            )["registry_to_supplied_role"]
            put(cfg["family"], {**e, "mapping": alternative})
    from forge.corpus import compose_lipid_miao_primary as miao

    for label, path, loader in [
        (
            "miao_cyclic",
            "results/phase1/compose_lipid_miao_cyclic_source_v2/replay-config.json",
            fixed,
        ),
        (
            "miao_primary",
            read(cfg_path("results/phase1/compose_lipid_miao_primary_origins_v1/config.json"))[
                "contracts"
            ]["miao"]["path"],
            miao,
        ),
    ]:
        cfg, _, local, checked = loader.load_contract(MIRROR, cfg_path(path))
        controls[label] = checked
        for family, e in local.items():
            put(family, full_fixed(e, bound=cfg["maximum_outcomes"]))
    cfg, _, variant = passerini._load(
        MIRROR, cfg_path("configs/multireaction/compose_lipid_v8_passerini_program_v1.json")
    )
    p = read(cfg_path("configs/multireaction/compose_lipid_family_replay_v1.json"))["inputs"][
        "passerini_registry"
    ]
    path = resolve_pin(p, MIRROR, label="Passerini registry")
    assert read(path)["reactions"] == [variant]
    adapter = RegistryAssemblyAdapter.from_registry(
        path, reaction_id=variant["reaction_id"], expected_sha256=p["sha256"]
    )
    put(
        cfg["family"],
        full_fixed(
            dict(
                run=partial(_passerini_replay, adapter, cfg["source_contract"]),
                mapping=cfg["source_contract"]["registry_to_source_roles"],
                source_contract=cfg["source_contract"],
            )
        ),
    )
    controls["passerini"] = passerini._controls(adapter, cfg)
    for label, path, loader in [
        (
            "thiol_yne",
            "results/phase1/compose_lipid_thiol_yne_source_v1/replay-config.json",
            staged,
        ),
        ("han", "results/phase1/compose_lipid_han_db_source_v1/replay-config.json", grouped),
        (
            "acid_epoxide",
            "results/phase1/compose_lipid_acid_epoxide_source_v2/replay-config.json",
            staged,
        ),
    ]:
        cfg, _, local, checked = loader.load_contract(MIRROR, cfg_path(path))
        controls[label] = checked
        for family, e in local.items():
            mapping = e.get("mapping") or {
                r: s for s, roles in e["occurrences"].items() for r in roles
            }
            put(
                family,
                {**e, "kind": "grouped" if loader is grouped else "staged", "mapping": mapping},
            )
    from forge.corpus import compose_lipid_ester_thiol as ester_thiol

    path = read(cfg_path("results/phase1/compose_lipid_ester_thiol_origins_v1/config.json"))[
        "contracts"
    ]["ester_thiol"]
    cfg, program, domain, checked = ester_thiol.load_contract(
        MIRROR, resolve_pin(path, MIRROR, label="ester thiol")
    )
    controls["ester_thiol"] = checked
    mapping = {r: s for s, roles in cfg["source_role_occurrences"].items() for r in roles}
    put(
        "preassembled_thiol_yne_tail_amidation",
        dict(
            kind="program",
            program=program,
            mapping=mapping,
            domain=domain,
            domain_kind="ester_thiol",
            domain_roles=[
                r for r in program.roles if r in cfg["source_role_occurrences"]["thiol_tail"]
            ],
        ),
    )
    cfg = read(cfg_path("configs/multireaction/compose_lipid_v8_staar_program_v1.json"))
    path = resolve_pin(cfg["inputs"]["registry"], MIRROR, label="STAAR")
    for family, binding in cfg["families"].items():
        program = RegistrySequentialProgram.from_registry(
            path,
            program_id=binding["program_id"],
            expected_sha256=cfg["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**cfg["search_bounds"]),
        )
        put(family, dict(kind="program", program=program, mapping={r: r for r in program.roles}))
    doc = read(resolve_pin(cfg["inputs"]["adjudication"], MIRROR, label="STAAR controls"))
    for name, asset in doc["assets"].items():
        resolve_pin(asset, MIRROR, label=name)
    doc = read(
        resolve_pin(
            doc["assets"]["control_transcriptions.json"],
            MIRROR,
            label="STAAR control transcriptions",
        )
    )
    controls["staar"] = {}
    for c in doc["controls"]:
        checked = program.replay(c["components"], c["expected_product"])
        assert checked["computed_consistency_pass"] is True
        controls["staar"][c["label"]] = checked
    # Extra source domains stay attached to the same executor used by preparation.
    from forge.corpus import compose_lipid_aema_ester as aema_ester
    from forge.corpus import compose_lipid_aema_replay as aema
    from forge.corpus import compose_lipid_ren_love_replay as ren
    from forge.corpus import compose_lipid_zhou_replay as zhou

    for label, path, loader in [
        ("aema", "results/phase1/compose_lipid_aema_source_v1/replay-config.json", aema),
        (
            "aema_ester",
            "results/phase1/compose_lipid_fast_qualification_v1/aema-config.json",
            aema_ester,
        ),
        ("zhou", "results/phase1/compose_lipid_user_supplements_v1/zhou-replay-config.json", zhou),
        ("ren", "results/phase1/compose_lipid_user_supplements_v1/replay-config.json", ren),
    ]:
        cfg, _, local, checked = loader.load_contract(MIRROR, cfg_path(path))
        controls[label] = checked
        families = local if loader is ren else {"aema_aza_thiol_addition": local}
        for family, by_events in families.items():
            for e in by_events.values():
                key = (
                    family
                    + ":"
                    + digest(
                        e["mapping"]
                        if loader is aema
                        else {
                            "mapping": e["mapping"],
                            "registry": cfg["inputs"]["registry"]["sha256"],
                        }
                    )
                )
                put(key, dict(**e, kind="grouped", quantities=e["program"].quantities))
    for form, p in read(cfg_path("results/phase1/compose_lipid_iphos_origins_v2/config.json"))[
        "contracts"
    ].items():
        cfg, _, adapters, programs, bounds, checked = current.load_contract(
            MIRROR, resolve_pin(p, MIRROR, label=form)
        )
        controls["iphos_" + form] = checked
        for family, adapter in adapters.items():
            put(
                family + ":" + form,
                dict(
                    kind="repeated",
                    adapter=adapter,
                    program=programs[family],
                    bounds=bounds,
                    mapping=cfg["families"][family]["registry_to_source_roles"],
                ),
            )
    cfg_path("results/phase1/compose_lipid_alias_origins_v1/run.py")
    return found, inputs, controls


def matching(found, layout):
    key = layout.record.program_id
    options = found.get(key, found.get(layout.family, []))
    if not options and ":source_roles:" in key:
        options = found.get(key.split(":source_roles:")[0], [])
    candidates = []
    for e in options:
        if ":source_roles:" in key and e["kind"] == "repeated":
            # The frozen alias-origin builder encodes names in adapter.roles
            # order (compose_lipid_alias_origins_v1/run.py::program_key).
            names = key.split(":source_roles:", 1)[1].split("/")
            if len(names) != len(e["adapter"].roles) or len(set(names)) != len(names):
                raise ValueError("Malformed qualified source-role namespace")
            e = {**e, "mapping": dict(zip(e["adapter"].roles, names, strict=True))}
        mapping = e["mapping"]
        if set(mapping.values()) != set(layout.quantities):
            continue
        quantities = e.get("quantities")
        if (
            quantities is not None
            and {mapping[r]: q for r, q in quantities.items()} != layout.quantities
        ):
            continue
        # Fixed events declare one complete component per source role.
        if e["kind"] == "fixed" and any(q != 1 for q in layout.quantities.values()):
            continue
        candidates.append(e)
    return candidates


def assess(found, layout, smiles):
    return assess_product(matching(found, layout), smiles, events=layout.record.program_depth)
