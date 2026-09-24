"""Retrieve additional original chemistry assets from author/publisher repositories."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCES={
 '22902058/li-thiol-yne-main.pdf':'https://www.levkingroup.com/images/pdf/Linxian_Li_Lipidoids_for_Gene-Delivery_1_Biomaterials_2012.pdf',
 '39099464/rsc-si.pdf':'https://pubs.rsc.org/suppdata/d4/tb/d4tb00960f/d4tb00960f1.pdf',
 '26729861/aema-si.pdf':'https://pmc.ncbi.nlm.nih.gov/articles/PMC4725465/bin/pnas.1520756113.sapp.pdf',
}


def get(item):
 name,url=item
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'FORGE source evidence audit; public original supplements'})
  with urllib.request.urlopen(request,timeout=45) as r:data=r.read()
  if not data.startswith(b'%PDF'):raise ValueError('Response is not a PDF')
  path=HERE/'remaining'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  return {'url':url,'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'retrieved'}
 except Exception as e:return {'url':url,'status':'unavailable','error':str(e)}


if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(get,SOURCES.items()))
 out={'date':'2026-09-20','seed':0,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records}
 (HERE/'remaining/next-primary-acquisition.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(records))
