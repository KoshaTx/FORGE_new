"""Retrieve original iPhos manuscript and publisher figure without changing source data."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    'article.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8188687/fullTextXML',
    'fig1.png': 'https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41563-020-00886-0/MediaObjects/41563_2020_886_Fig1_HTML.png',
}

def fetch(item):
    name,url=item
    try:
        with urllib.request.urlopen(url,timeout=45) as r: data=r.read()
        if (name.endswith('.png') and not data.startswith(b'\x89PNG')) or (name.endswith('.xml') and b'<article' not in data):
            raise ValueError('Unexpected response format')
        path=HERE/name;path.write_bytes(data)
        return {'url':url,'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'status':'retrieved'}
    except Exception as exc:
        return {'url':url,'status':'unavailable','error':str(exc)}

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool: records=list(pool.map(fetch,SOURCES.items()))
    (HERE/'acquisition.json').write_text(json.dumps({'date':'2026-09-20','seed':0,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records},indent=2,sort_keys=True)+'\n')
    print(json.dumps(records))
