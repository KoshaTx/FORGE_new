"""Try canonical publisher and PMC instance paths for the original missing supplements."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCES={
 '26729861/aema-publisher-si.pdf':'https://www.pnas.org/doi/suppl/10.1073/pnas.1520756113/suppl_file/pnas.1520756113.sapp.pdf',
 '26729861/aema-instance-si.pdf':'https://pmc.ncbi.nlm.nih.gov/articles/instance/4725465/bin/pnas.1520756113.sapp.pdf',
 '20080679/epoxide-publisher-si.pdf':'https://www.pnas.org/doi/suppl/10.1073/pnas.0910603106/suppl_file/pnas.0910603106_SI.pdf',
 '20080679/epoxide-instance-si.pdf':'https://pmc.ncbi.nlm.nih.gov/articles/instance/2804742/bin/0910603106_pnas.0910603106_SI.pdf',
}


def get(item):
 name,url=item
 try:
  path=HERE/'remaining'/name
  if path.exists():raise ValueError('Refusing to overwrite a prior asset')
  request=urllib.request.Request(url,headers={'User-Agent':'FORGE original chemistry supplement audit'})
  with urllib.request.urlopen(request,timeout=40) as r:
   content_type=r.headers.get('Content-Type');data=r.read()
  if not data.startswith(b'%PDF'):raise ValueError('Response is not a PDF: '+str(content_type))
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  return {'url':url,'path':str(path.relative_to(HERE.parents[2])),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'retrieved'}
 except Exception as e:return {'url':url,'status':'unavailable','error':str(e)}


if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(get,SOURCES.items()))
 out={'date':'2026-09-20','seed':0,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records}
 (HERE/'remaining/pnas-canonical-acquisition.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(records))
