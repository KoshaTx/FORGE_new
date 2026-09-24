"""Inventory every family and the actual remaining preparation gates; never emit training data."""
import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def main():
    readiness_path=ROOT/'results/phase1/compose_lipid_readiness_extension_v1/result.json'
    source_path=ROOT/'results/phase1/compose_lipid_all_family_sources_v1/remaining/result.json'
    key_path=ROOT/'data/source_cache/compose_lipid_supplement_2026-09-19/family_decomposition_key.json'
    r=json.loads(readiness_path.read_text());key=json.loads(key_path.read_text());sources=json.loads(source_path.read_text())
    source_by_id={x['pmid']:x for x in sources['records']}
    all_pdf={}
    for pmid,record in source_by_id.items():
        parent=source_path.parent/pmid
        all_pdf[pmid]=[pin(ROOT,p) for p in sorted(parent.glob('*.pdf'))]
        for asset in record['assets'].values():
            resolve_pin({k:asset[k] for k in ['path','sha256']},ROOT,label=pmid)
    details={
        'a3_amine_aldehyde_alkyne':'Resolve 1,214 site-ambiguous recipes and the rejected Han ordered-two-event architecture description; do not select a site using the target.',
        'o_esterification':'Recover O-selective source evidence for heads with competing primary/secondary amines, or keep those 909 rows outside this program scope.',
        'aryl_reductive_amination':'Resolve remaining site/role tuple ambiguity with exact precursor and attachment evidence.',
        'aza_michael_acrylate':'Resolve source role tuples, occupancy and any remaining site ambiguity without collapsing complete acceptors.',
        'acid_epoxide_diester_multistep':'Original 5,279 eligible recipes are authenticated; inspect the missing original synthesis SI, then qualify acid-epoxide opening and O-acylation stages with atom origin and stage controls.',
        'aema_aza_thiol_addition':'Qualify the ordered aza/thiol additions and preserve complete AEMA objects, role multiplicity and event order.',
        'alpha_isocyanoester_dihydroimidazole':'Visually adjudicate Miao SI cyclic primary-amine branch and ring-closure atom origins; keep it separate from the secondary-amine acyclic branch.',
        'amine_alkylation':'Recover Ren SI and qualify N-alkylation with explicit N-H, charge and two-electrophile occupancy; do not silently allow tertiary-N quaternization.',
        'amine_epoxide_opening':'Adjudicate exact amine/epoxide sites and repeated occupancy; distinguish terminal/regioselective source scope from ambiguous ring opening.',
        'epoxide_opening_o_acylation':'Adjudicate recovered branched-lipidoid SI; qualify two-stage opening/acylation, individual alcohol sites and complete tail objects.',
        'iphos_ring_opening':'Adjudicate recovered iPhos SI; establish the cyclic phosphate opening graph, charge, complete coupled tails and source-specific amine sites.',
        'ketone_isocyanide_amide':'Adjudicate Miao SI acyclic secondary-amine branch, including carbonyl atom origin and exclusion of heterocyclic closure.',
        'ketone_ugi4':'Adjudicate recovered LiON SI; qualify the ketone Ugi-4 graph, acid role modes, retained ionizable nitrogen and indivisible coupled ketone arms.',
        'preassembled_thiol_yne_tail_amidation':'Recover original thiol-yne-tail preparation and final amidation evidence; qualify complete preassembled tails without treating their internal construction as one unresolved tail slot.',
        'vitamin_b5_multistep':'Adjudicate recovered vitamin-B5 SI; distinguish I7/I8/I9 stage order, primary/secondary O-acyl sites, head ester/amide and preserved source stereochemical provenance.'
    }
    families={}
    for family,v in r['summary']['by_family'].items():
        pdfs=[];publications=[]
        for publication in key.get(family,{}).get('literature_libraries',[]):
            pmid=publication.get('pmid');record=source_by_id.get(pmid)
            publications.append({'pmid':pmid,'doi':publication.get('doi') or (record or {}).get('doi'),'source_acquisition_attempted':record is not None,'downloaded_supplement_pdfs':all_pdf.get(pmid,[])})
            pdfs+=all_pdf.get(pmid,[])
        families[family]={**v,'formal_family':family in key,'training_ready':False,'next_program_work':details.get(family,'Current eligible recipes have exact computed replay; full-universe partition, remaining source scope, final representation and training admission are still required.' if family in key else 'Reference category; no qualified reaction program assigned.'),'source_reaction_invariant':key.get(family,{}).get('reaction_program_invariant'),'publications':publications,'unreviewed_new_pdf_count':len(pdfs)}
    total=r['summary']
    doc={'schema_version':'forge.all_family_training_preparation_worklist.v1','seed':0,'date':'2026-09-20','inputs':{'readiness':pin(ROOT,readiness_path),'family_definitions':pin(ROOT,key_path),'source_acquisition':pin(ROOT,source_path),'split_inputs':pin(ROOT,HERE/'required-split-inputs.json')},'implementation':pin(ROOT,Path(__file__).resolve()),'summary':{'source_records':total['source_records'],'formal_families':len(key),'families_with_exact_computed_evidence':sum(v['exact_computed_reconstructions']>0 for v in families.values()),'exact_computed_reconstructions':total['exact_reconstructions'],'eligible_records_pending_reconstruction':sum(v['pending_eligible_reconstructions'] for v in families.values()),'formal_families_without_exact_reconstruction':sum(v['formal_family'] and v['exact_computed_reconstructions']==0 for v in families.values())},'families':families,'full_corpus_gates':r['remaining_gates'],'no_silent_record_or_size_reduction':True,'training_admitted':False,'training_calls':0,'next_required_external_input':'The three original split producer/config files in required-split-inputs.json; chemistry qualification continues independently where primary sources are available.'}
    dump(HERE/'all-family-worklist.json',doc)
    print(json.dumps(doc['summary'],indent=2))


if __name__=='__main__':main()
