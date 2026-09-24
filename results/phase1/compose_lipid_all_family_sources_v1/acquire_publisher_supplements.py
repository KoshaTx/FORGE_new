"""Retrieve primary publisher supplements, with failures preserved as acquisition evidence."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent/'remaining'
SOURCES={
'31570898/miao-si.pdf':'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41587-019-0247-3/MediaObjects/41587_2019_247_MOESM1_ESM.pdf',
'33542471/iphos-si.pdf':'https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41563-020-00886-0/MediaObjects/41563_2020_886_MOESM1_ESM.pdf',
'39658727/ketone-ugi4-si.pdf':'https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41587-024-02490-y/MediaObjects/41587_2024_2490_MOESM1_ESM.pdf',
}


def get(item):
 name,url=item;record={'url':url,'path':str((HERE/name).relative_to(ROOT))}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'FORGE source evidence audit'}),timeout=45) as response:
   payload=response.read();record.update(status=response.status,final_url=response.url)
  if not payload.startswith(b'%PDF'):raise ValueError('Not a PDF')
  (HERE/name).write_bytes(payload)
  record.update(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),source_execution_admitted=False)
 except Exception as e:record['error']=str(e)
 print(record,flush=True);return record


if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(get,SOURCES.items()))
 (HERE/'publisher-acquisition.json').write_text(json.dumps({'seed':0,'date':'2026-09-20','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,'training_calls':0},indent=2,sort_keys=True)+'\n')
