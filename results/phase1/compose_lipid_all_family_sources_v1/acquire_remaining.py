"""Resolve exact primary records and their openly distributed supplementary files."""
import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent/'remaining'
KEY=ROOT/'data/source_cache/compose_lipid_supplement_2026-09-19/family_decomposition_key.json'
FAMILIES=['acid_epoxide_diester_multistep','aema_aza_thiol_addition','alpha_isocyanoester_dihydroimidazole','amine_alkylation','amine_epoxide_opening','epoxide_opening_o_acylation','iphos_ring_opening','ketone_isocyanide_amide','ketone_ugi4','preassembled_thiol_yne_tail_amidation','vitamin_b5_multistep']


def fetch(url,path):
    req=urllib.request.Request(url,headers={'User-Agent':'FORGE primary chemistry evidence audit'})
    with urllib.request.urlopen(req,timeout=50) as response:
        payload=response.read()
        if path.suffix=='.zip' and not payload.startswith(b'PK'):raise ValueError('Not a ZIP')
        if path.suffix=='.json':json.loads(payload)
        if path.suffix=='.xml' and b'<article' not in payload[:5000]:raise ValueError('Not article XML')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(payload)
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'url':url}


def one(pmid):
    out={'pmid':pmid,'assets':{},'failures':[]}
    base=HERE/pmid
    try:
        url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode({'query':'EXT_ID:'+pmid+' AND SRC:MED','format':'json','resultType':'core'})
        out['assets']['metadata']=fetch(url,base/'metadata.json')
        matches=json.loads((base/'metadata.json').read_text())['resultList']['result']
        if len(matches)!=1 or matches[0]['id']!=pmid:raise ValueError('Publication ID did not resolve exactly')
        record=matches[0];out.update(doi=record.get('doi'),pmcid=record.get('pmcid'),title=record['title'])
        if record.get('pmcid'):
            for name,endpoint in [('article.xml','fullTextXML'),('supplements.zip','supplementaryFiles')]:
                try:out['assets'][name]=fetch('https://www.ebi.ac.uk/europepmc/webservices/rest/'+record['pmcid']+'/'+endpoint,base/name)
                except Exception as e:out['failures'].append({'asset':name,'error':str(e)})
    except Exception as e:out['failures'].append({'asset':'metadata','error':str(e)})
    print(pmid, sorted(out['assets']), out['failures'],flush=True)
    return out


if __name__=='__main__':
    key=json.loads(KEY.read_text()); ids=sorted({source['pmid'] for f in FAMILIES for source in key[f]['literature_libraries'] if source.get('pmid')})
    HERE.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(one,ids))
    receipt={'schema_version':'forge.primary_source_acquisition.v1','seed':0,'date':'2026-09-20','inputs':{'family_key':{'path':str(KEY.relative_to(ROOT)),'sha256':hashlib.sha256(KEY.read_bytes()).hexdigest()},'script':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}},'records':records,'source_execution_admitted':False,'training_calls':0}
    (HERE/'result.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
