"""Reproduce the frozen threshold decisions and p=1 prediction counts."""
import argparse,collections,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('results',type=Path);p.add_argument('analysis',type=Path);p.add_argument('freeze',type=Path);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
freeze=json.loads(a.freeze.read_text());margin=freeze['author_decisions']['original_tolerance']
assert margin==.01 and freeze['author_decisions']['no_change_after_test']
d=json.loads((a.analysis/'analysis.json').read_text());assert d['scope']=='AI-executed frozen final-test evaluation'
counts=collections.Counter();n=collections.Counter()
for line in (a.results/'predictions.jsonl').open():
 r=json.loads(line)
 if r['deletion_probability']==1:
  k=r['seed'],r['feature'],r['augmented'];counts[k]+=r['prediction'];n[k]+=1
assert len(n)==12 and set(n.values())=={2000}
distribution=[dict(seed=k[0],feature=k[1],augmented=k[2],positive=counts[k],negative=n[k]-counts[k],n=n[k]) for k in n]
(a.out/'prediction_distribution.json').write_text(json.dumps(distribution,indent=2))
rows=[]
for f in ['word','char']:
 orig=d['comparisons'][f'h2_{f}_original_change']['ci95_percentile'][0]
 gain=d['comparisons'][f'h2_{f}_perturbed_gain']['ci95_percentile'][0]
 rows.append(dict(feature=f,original_ci_lower=orig,perturbed_gain_ci_lower=gain,within_001=orig>-.01,within_0005=orig>-.005,h2_supported=orig>-.01 and gain>0))
(a.out/'threshold_decisions.json').write_text(json.dumps(rows,indent=2))
print('Saved frozen decisions and12 prediction-count rows.')
