"""Measure zero-vector inputs with the exact training subsets and TF-IDF configuration."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from run_experiment import load,corrupt

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();assert not a.out.exists()
meta=json.loads((a.results/'metadata.json').read_text());split=json.loads((a.results/'split_ids.json').read_text())
rows={r[0]:r for r in load(a.data_dir/'ratings_train.txt')};evaluation=[rows[i] for i in split['evaluation']]
res=[]
for seed,ids in split['training'].items():
 train=[rows[i] for i in ids];seed=int(seed)
 for feature in ['word','char']:
  for aug in [False,True]:
   texts=[r[1] for r in train]
   if aug:texts += [corrupt(r[1],.5,seed+1000,r[0]) for r in train]
   common=dict(lowercase=False,min_df=2,max_features=meta['args']['max_features'],sublinear_tf=True)
   v=TfidfVectorizer(tokenizer=str.split,token_pattern=None,ngram_range=(1,2),**common) if feature=='word' else TfidfVectorizer(analyzer='char',ngram_range=(2,5),**common)
   v.fit(texts)
   for rate in [0.,.25,.5,1.]:
    x=v.transform([corrupt(r[1],rate,9000,r[0]) for r in evaluation])
    res.append(dict(seed=seed,feature=feature,augmented=aug,rate=rate,zero_vector_count=int(np.count_nonzero(x.getnnz(axis=1)==0)),n_eval=len(evaluation)))
a.out.write_text(json.dumps(res,indent=2));print('Saved',len(res),'feature-coverage conditions')
lines=['# TF-IDF zero-vector proportions\n','Development inputs only. Values average the proportion over the three training seeds.\n','| Feature | Augmented | p=0 | p=.25 | p=.5 | p=1 |','|---|---|---:|---:|---:|---:|']
for feature in ['word','char']:
 for aug in [False,True]:
  values=[np.mean([r['zero_vector_count']/r['n_eval'] for r in res if r['feature']==feature and r['augmented']==aug and r['rate']==rate]) for rate in [0.,.25,.5,1.]]
  lines.append('| '+feature+' | '+str(aug)+' | '+' | '.join(f'{v:.1%}' for v in values)+' |')
a.out.with_suffix('.md').write_text('\n'.join(lines))
