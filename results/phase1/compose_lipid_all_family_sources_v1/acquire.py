"""Retrieve primary publisher/source material; retain bytes and acquisition receipts."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCES = {
    'amine_alkylation_si.pdf': 'https://www.rsc.org/suppdata/d4/tb/d4tb00960f/d4tb00960f1.pdf',
    'maleate_article.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11888472/fullTextXML',
    'maleate_supplements.zip': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11888472/supplementaryFiles',
    'o_esterification_figshare.json': 'https://api.figshare.com/v2/articles?resource_doi=10.1021/acsnano.2c07822',
}


def acquire(item):
    name, url = item
    record = {'name': name, 'url': url}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'FORGE source evidence audit'})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
            record.update(status=response.status, final_url=response.url, content_type=response.headers.get('Content-Type'))
        if name.endswith('.pdf') and not payload.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF')
        if name.endswith('.zip') and not payload.startswith(b'PK'):
            raise ValueError('Response is not a ZIP')
        if name.endswith('.json'):
            json.loads(payload)
        (OUT / name).write_bytes(payload)
        record.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    except Exception as error:
        record['error'] = str(error)
    return record


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(acquire, SOURCES.items()))
    receipt = {'seed': 0, 'date': '2026-09-20', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': records}
    (OUT / 'acquisition.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
