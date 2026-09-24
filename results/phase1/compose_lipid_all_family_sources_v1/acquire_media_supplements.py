"""Use the publisher's current media host for two previously unresolved supplements."""
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('publisher', HERE/'acquire_publisher_supplements.py')
publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
sources={k:v.replace('https://static-content.springer-cdn.com/esm/','https://media.springernature.com/original/springer-static/esm/') for k,v in publisher.SOURCES.items() if k.startswith(('33542471/','39658727/'))}
with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(publisher.get,sources.items()))
(HERE/'remaining/media-acquisition.json').write_text(json.dumps({'seed':0,'date':'2026-09-20','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parent_script_sha256':hashlib.sha256((HERE/'acquire_publisher_supplements.py').read_bytes()).hexdigest(),'records':records,'training_calls':0},indent=2,sort_keys=True)+'\n')
