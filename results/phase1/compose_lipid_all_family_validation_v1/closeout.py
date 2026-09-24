"""Close this qualification stage against pinned science and full validation receipts."""
import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def main():
    inputs={
        'readiness':'results/phase1/compose_lipid_readiness_extension_v1/result.json',
        'worklist':'results/phase1/compose_lipid_training_readiness_v1/all-family-worklist.json',
        'split_inputs':'results/phase1/compose_lipid_training_readiness_v1/required-split-inputs.json',
        'validation':'results/phase1/compose_lipid_all_family_validation_v1/validation_report.json',
        'input_audit':'results/phase1/compose_lipid_all_family_validation_v1/input-pin-audit.json',
        'maleate_ester':'results/phase1/compose_lipid_supplied_maleate_ester_v1/result.json',
        'thiol_a3':'results/phase1/compose_lipid_supplied_thiol_a3_v1/result.json',
    }
    paths={k:ROOT/v for k,v in inputs.items()}
    docs={k:json.loads(v.read_text()) for k,v in paths.items()}
    for k in ['readiness','maleate_ester','thiol_a3']:
        resolve_pin(docs[k]['artifact'],ROOT,label=k+' ledger')
    validate=docs['validation'];work=docs['worklist']
    if validate['new_failures'] or validate['new_errors'] or not validate['source_snapshot_unchanged'] or validate['vendor_verify_exit_code'] or validate['focused']['failures'] or validate['focused']['errors']:
        raise ValueError('New regression, source mutation or failed focused/vendor check requires resolution')
    for path,h in docs['input_audit']['checked_files'].items():
        resolve_pin({'path':path,'sha256':h},ROOT,label='input audit')
    result={
        'schema_version':'forge.current_all_family_qualification_closeout.v1','seed':0,'date':'2026-09-20',
        'status':'additional_computed_reconstructions_training_unqualified',
        'inputs':{k:pin(ROOT,p) for k,p in paths.items()},'implementation':pin(ROOT,Path(__file__).resolve()),
        'summary':{**work['summary'],'new_exact_reconstructions':docs['readiness']['summary']['new_exact_reconstructions']},
        'validation':{k:validate[k] for k in ['focused','full','new_failures','new_errors','resolved_failures','resolved_errors','source_snapshot_unchanged','vendor_verify_exit_code']},
        'full_universe_partition_requires_original_split_producers':True,
        'remaining_gates':docs['readiness']['remaining_gates'],
        'phase1_definition_of_done_met':False,'training_ready':False,'training_calls':0,'training_admitted':False,
        'budgeted_remote_compute_launched':False,
    }
    dump(HERE/'result.json',result)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
