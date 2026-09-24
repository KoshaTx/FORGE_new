"""Preserve negative scope results alongside successful source-qualified replays."""
import gzip
import json
from collections import Counter
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def ledger(name):
    result_path=ROOT/'results/phase1'/name/'result.json'
    result=json.loads(result_path.read_text())
    path=resolve_pin(result['artifact'],ROOT,label=name)
    with gzip.open(path,'rt') as f:
        records=[json.loads(line) for line in f]
    return result_path,result,records


def main():
    p,old,miao=ledger('compose_lipid_supplied_miao_cyclic_v1')
    p2,new,miao_new=ledger('compose_lipid_supplied_miao_cyclic_v2')
    p3,thiol,thiol_rows=ledger('compose_lipid_supplied_thiol_yne_v1')
    isotope=Counter(c[1] for row in miao for c in row['component_instances'] if c[0]=='isocyanide')
    failures=Counter()
    for row in thiol_rows:
        if row['replay']['computed_consistency_pass']: continue
        for role,detail in row['replay']['terminal_constraints'].items():
            failures.update(role+':'+key for key,passed in detail['checks'].items() if not passed)
    catalog=ROOT/'data/source_cache/compose_lipid_supplement_2026-09-19/precursor_catalog.jsonl.gz'
    selected={}
    with gzip.open(catalog,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['component_id'] in isotope:
                selected[row['component_id']]=row['constitution']
    dump(HERE/'source-scope-audit.json',{
        'schema_version':'forge.compose_lipid_source_scope_audit.v1','seed':0,
        'implementation':pin(ROOT,Path(__file__).resolve()),
        'inputs':{'miao_initial':pin(ROOT,p),'miao_drawn_precursors':pin(ROOT,p2),'thiol_yne':pin(ROOT,p3),'catalog':pin(ROOT,catalog)},
        'miao_initial':{'rows':len(miao),'exact':sum(r['replay']['computed_consistency_pass'] for r in miao),'isocyanide_ids':dict(isotope),'complete_precursors':selected,'disposition':'Initial isolated Iso5 ethyl scope does not include the corpus Iso6 tert-butyl precursor; zero is a preserved valid negative result.'},
        'miao_source_library_extension':{'rows':len(miao_new),'exact':sum(r['replay']['computed_consistency_pass'] for r in miao_new),'disposition':'Primary Fig. 1b generic cyclic ester plus explicit Fig. 1c Iso4/5/6 support the bounded computed net graph. Isolated characterization remains Iso5 only.'},
        'thiol_yne':{'rows':len(thiol_rows),'exact':sum(r['replay']['computed_consistency_pass'] for r in thiol_rows),'unresolved':sum(not r['replay']['computed_consistency_pass'] for r in thiol_rows),'failed_terminal_constraints':dict(failures),'disposition':'Every row reconstructs the supplied source graph, but out-of-scope precursor atoms remain excluded. Source alkyl-thiol scope is not broadened just because a product matches.'},
        'training_admitted':False,'experimental_execution_admitted':False,
    })


if __name__=='__main__':main()
