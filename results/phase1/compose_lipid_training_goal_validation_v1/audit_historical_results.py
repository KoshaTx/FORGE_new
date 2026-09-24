"""Read-only audit of missing pinned result files against the original local checkout."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = Path('/Users/rahulmaganti/Kosha/forge')
HERE = Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def pins(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            yield value['path'],value['sha256']
        for child in value.values():yield from pins(child)
    elif isinstance(value,list):
        for child in value:yield from pins(child)


def main():
    references=defaultdict(set);inputs={}
    for path in sorted((ROOT/'configs').rglob('*.json')):
        doc=json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))]=sha(path)
        for relative,digest in pins(doc):
            if relative.startswith('results/') and not (ROOT/relative).is_file():
                references[(relative,digest)].add(str(path.relative_to(ROOT)))
    records=[]
    for (relative,expected),configs in sorted(references.items()):
        old=OLD/relative
        observed=sha(old) if old.is_file() else None
        records.append({'path':relative,'expected_sha256':expected,'old_checkout_sha256':observed,
                        'disposition':'exact_original_bytes_available' if observed==expected else 'missing_original' if observed is None else 'different_version',
                        'referencing_configs':sorted(configs)})
    counts=defaultdict(int)
    for r in records:counts[r['disposition']]+=1
    out={'schema_version':'forge.historical_result_recovery_audit.v1','seed':0,'script_sha256':sha(Path(__file__)),
         'input_config_sha256s':inputs,'original_checkout':str(OLD),'summary':dict(counts),'records':records,
         'mutation_performed':False,'historical_results_executed':False,'sealed_molecular_payloads_decoded':False}
    (HERE/'historical-result-recovery-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out['summary']))


if __name__=='__main__':main()
