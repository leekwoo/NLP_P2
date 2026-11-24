"""Compare a reproduction against the original experiment, excluding runtime."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path);p.add_argument('reproduction',type=Path)
a=p.parse_args()
x=json.loads((a.reference/'metrics.json').read_text());y=json.loads((a.reproduction/'metrics.json').read_text())
assert len(x)==len(y)==48
for r,s in zip(x,y):
 assert r.keys()==s.keys()
 for k in r:
  if k=='fit_seconds':continue
  assert (abs(r[k]-s[k])<1e-10 if isinstance(r[k],float) else r[k]==s[k]),(k,r,s)
assert json.loads((a.reference/'split_ids.json').read_text())==json.loads((a.reproduction/'split_ids.json').read_text())
m=[json.loads((folder/'metadata.json').read_text()) for folder in [a.reference,a.reproduction]]
for k in ['data_sha256','script_sha256','started_scope','dataset_commit']:
 assert m[0][k]==m[1][k],k
if all((folder/'predictions.jsonl').exists() for folder in [a.reference,a.reproduction]):
 def sha(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  return h.hexdigest()
 assert sha(a.reference/'predictions.jsonl')==sha(a.reproduction/'predictions.jsonl')
 print('All prediction records match byte for byte.')
print('48 metric rows, split IDs, input/code hashes and evaluation scope match.')
