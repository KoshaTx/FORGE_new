"""Explain failed qualified scopes from supplied eligible components and unfiltered replay."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump,pin
from forge.corpus.compose_lipid_supplement import rows
from forge.core.hashing import resolve_pin

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def main():
    result_paths={
        'maleate_ester':ROOT/'results/phase1/compose_lipid_supplied_maleate_ester_v1/result.json',
        'thiol_a3':ROOT/'results/phase1/compose_lipid_supplied_thiol_a3_v1/result.json',
    }
    precursor=ROOT/'data/source_cache/compose_lipid_supplement_2026-09-19/precursor_catalog.jsonl.gz'
    structures={r['component_id']:r['constitution'] for r in rows(precursor)}
    groups=defaultdict(Counter)
    for name,path in result_paths.items():
        result=json.loads(path.read_text()); ledger=resolve_pin(result['artifact'],ROOT,label=name)
        for row in rows(ledger):
            replay=row['replay']
            if replay['computed_consistency_pass']:continue
            family=row['family']
            if family=='o_esterification':
                head=next(i for role,i,n in row['component_instances'] if role=='aminoalcohol_head')
                groups['o_esterification_unresolved_heads'][structures[head]]+=1
            elif family=='a3_amine_aldehyde_alkyne':
                groups['a3_competing_forward_product_counts'][str(len(replay['forward_layers'][-1]))]+=1
                for k,v in replay['checks'].items():groups['a3_failed_checks'][k]+=not v
            elif family=='maleate_addition' and name=='maleate_ester':
                head=next(i for role,i,n in row['component_instances'] if role=='amine_head')
                groups['maleate_amine_program_unresolved_heads'][structures[head]]+=1
    dump(HERE/'unresolved-scope-analysis.json',{'schema_version':'forge.source_scope_failure_analysis.v1','seed':0,'inputs':{**{k:pin(ROOT,v) for k,v in result_paths.items()},'precursors':pin(ROOT,precursor)},'implementation':pin(ROOT,Path(__file__).resolve()),'counts':{k:dict(v) for k,v in groups.items()},'dispositions':{'maleate':'All 1,285 amine-program misses have source monothiol heads; separate thiol program closes them.','o_esterification':'These four head structures have free NH sites. O-selective source support is not qualified for those competitors.','a3':'Keep all competing final constitutional products; do not use the target to choose an attachment site.'},'training_admitted':False,'training_calls':0})


if __name__=='__main__':main()
