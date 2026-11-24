"""Verify predictions, summarize development evidence, and prepare blinded human review."""
import argparse, json, random
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

SEEDS=[42,43,44]
FEATURES=['word','char']
RATES=[0.0,0.25,0.5,1.0]

def metrics_from_counts(tp,tn,fp,fn):
    d1=2*tp+fp+fn; d0=2*tn+fp+fn
    return (np.divide(2*tp,d1,out=np.zeros_like(d1,dtype=float),where=d1!=0)+np.divide(2*tn,d0,out=np.zeros_like(d0,dtype=float),where=d0!=0))/2

def contrasts(scores):
    # scores shape seed, feature, augmentation, deletion rate
    avg=scores[:,:,:,1:].mean(axis=-1)
    drops=scores[:,:,:,0]-avg
    return {'h1_word_drop_minus_char_drop':float((drops[:,0,0]-drops[:,1,0]).mean()),
            **{f'h2_{f}_perturbed_gain':float((avg[:,i,1]-avg[:,i,0]).mean()) for i,f in enumerate(FEATURES)},
            **{f'h2_{f}_original_change':float((scores[:,i,1,0]-scores[:,i,0,0]).mean()) for i,f in enumerate(FEATURES)}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--final-test',action='store_true',help='Analyze frozen final-test results; do not create a new manual-review sample.')
    ap.add_argument('--bootstrap',type=int,default=1000);ap.add_argument('--bootstrap-seed',type=int,default=11042026)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    metrics=json.loads((a.results/'metrics.json').read_text())
    meta=json.loads((a.results/'metadata.json').read_text())
    assert meta['started_scope']==('final_test' if a.final_test else 'development_pilot'),'Requested and recorded evaluation scopes differ.'
    ids=json.loads((a.results/'split_ids.json').read_text())['evaluation'];pos={v:i for i,v in enumerate(ids)}
    shape=(3,2,2,4,len(ids));pred=np.full(shape,-1,dtype=np.int8); labels=np.full(len(ids),-1,dtype=np.int8)
    original={};variants={}
    with (a.results/'predictions.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line);i=pos[r['id']];k=(SEEDS.index(r['seed']),FEATURES.index(r['feature']),int(r['augmented']),RATES.index(r['deletion_probability']),i)
            assert pred[k]==-1,'Duplicate prediction'
            pred[k]=r['prediction']
            assert labels[i] in (-1,r['label']);labels[i]=r['label']
            assert original.setdefault(r['id'],r['original'])==r['original']
            assert variants.setdefault((r['id'],r['deletion_probability']),r['perturbed'])==r['perturbed'],'Evaluation corruption differs across models'
    assert (pred>=0).all() and set(labels)=={0,1}
    score=np.empty(shape[:-1])
    for ix in np.ndindex(score.shape): score[ix]=f1_score(labels,pred[ix],average='macro',zero_division=0)
    for r in metrics:
        ix=(SEEDS.index(r['seed']),FEATURES.index(r['feature']),int(r['augmented']),RATES.index(r['deletion_probability']))
        assert abs(score[ix]-r['macro_f1'])<1e-12
    assert np.array_equal(pred[...,0,:].shape,pred[...,1,:].shape)
    flat=pred.reshape(48,len(ids));tp=((flat==1)&(labels==1)).astype(float);tn=((flat==0)&(labels==0)).astype(float)
    fp=((flat==1)&(labels==0)).astype(float);fn=((flat==0)&(labels==1)).astype(float)
    rng=np.random.default_rng(a.bootstrap_seed);values={k:[] for k in contrasts(score)}
    # Same review resample applied to every model, rate and seed. Preserve class prevalence variability.
    for start in range(0,a.bootstrap,100):
        count=min(100,a.bootstrap-start)
        weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=count).T
        batch=metrics_from_counts(tp@weights,tn@weights,fp@weights,fn@weights)
        for b in range(count):
            for k,v in contrasts(batch[:,b].reshape(score.shape)).items(): values[k].append(v)
    ci={k:{'estimate':v,'ci95_percentile':np.quantile(values[k],[.025,.975]).tolist()} for k,v in contrasts(score).items()}
    summary=[]
    for f,feature in enumerate(FEATURES):
        for aug in [0,1]:
            means=score[:,f,aug,:].mean(0);std=score[:,f,aug,:].std(0,ddof=1)
            noise=score[:,f,aug,1:].mean(-1);drop=score[:,f,aug,0]-noise
            summary.append(dict(feature=feature,augmented=bool(aug),means=means.tolist(),std=std.tolist(),mean_perturbed=float(noise.mean()),std_perturbed=float(noise.std(ddof=1)),mean_drop=float(drop.mean())))
    report={'scope':('AI-executed frozen final-test evaluation' if a.final_test else 'AI-executed exploratory development evaluation, not final test'),'summary':summary,'comparisons':ci,
            'bootstrap':{'replicates':a.bootstrap,'seed':a.bootstrap_seed,'unit':'review ID, paired across all models, rates and seeds','interval':'95% percentile, conditional on trained models, not corrected for multiple comparisons'},'verification':'All 96,000 saved predictions and 48 metric rows checked; identical evaluation texts across models/seeds.'}
    (a.out/'analysis.json').write_text(json.dumps(report,indent=2))
    lines=['# 확대 개발 실험 결과\n','이 결과는 AI가 기존 설정으로 실행한 **개발용 예비 결과**다. 최종 시험 데이터를 사용하지 않았으며, 학생의 본실험으로 표현하지 않는다.\n',
    '| 표현 | 증강 | 원문 | p=.25 | p=.50 | p=1 | 교란 평균 | 평균 하락 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        cells=[('어절' if row['feature']=='word' else '문자'),('있음' if row['augmented'] else '없음')]+[f'{v:.4f}' for v in row['means']]+[f"{row['mean_perturbed']:.4f}",f"{row['mean_drop']:.4f}"]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['\n표의 점수는 3개 seed의 macro-F1 평균이다. 교란 평균은 각 seed에서 세 삭제 확률의 macro-F1을 산술평균한 값이다. 원문은 수동 교정한 문장이 아니라 NSMC 원본 입력이다. 표준편차와 상세 값은 analysis.json에 있다.\n','| 비교 | 평균 차이 | 95% paired bootstrap 구간 |','|---|---:|---|']
    for k,v in ci.items(): lines.append(f"| {k} | {v['estimate']:.4f} | [{v['ci95_percentile'][0]:.4f}, {v['ci95_percentile'][1]:.4f}] |")
    lines+=['\nH1 차이는 어절 평균 하락 − 문자 평균 하락이다. 양수이면 H1 방향이다. H2 gain/change는 증강 후 − 증강 전이다. 0.01 원문 허용 기준은 여전히 미확정이며, 개발 결과를 본 뒤 확정하면 그 사실을 기록해야 한다.\n',f"개발 실행 전체 시간: {meta['total_seconds']:.2f}초. 12회 학습, 학습 원본 10,000개/seed, 개발 2,000개. Bootstrap {a.bootstrap}회, seed {a.bootstrap_seed}. 구간은 평가 표본 불확실성을 나타내며 모델 학습 전체의 불확실성을 포괄하지 않는다. 다중 비교 보정은 하지 않았다.\n"]
    if a.final_test:
        lines[0]='# 설정 고정 후 최종 시험 결과\n'
        lines[1]='학생이 기준을 확정한 뒤 AI가 고정 설정으로 실행한 최종 시험 결과다. 개발 결과와 구분한다.\n'
        lines=[line.replace('0.01 원문 허용 기준은 여전히 미확정이며, 개발 결과를 본 뒤 확정하면 그 사실을 기록해야 한다.', '학생은 개발 결과를 본 뒤 허용 기준0.01을 확정했다. 원문 변화95% 구간의 하한이 -0.01보다 클 때만 허용 범위 내로 판정한다. 사전 등록된 표준이 아니며 시험 결과를 보고 기준을 바꾸지 않는다.').replace('개발 실행 전체 시간:', '최종 시험 실행 전체 시간:').replace('개발 2,000개', '시험 2,000개') for line in lines]
    (a.out/'RESULTS.md').write_text('\n'.join(lines))
    if a.final_test:
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return
    # Select before inspecting correctness. Preserve pairing without displaying model predictions or gold labels.
    review_ids=random.Random(1104).sample(ids,50)
    review=['# 공백 삭제 후 감성 유지 여부: 학생 직접 검토\n','총 50개 리뷰 × 3개 변형 = 150쌍. 모델 정답 여부와 무관하게 고정 추출(seed 1104)했다. 정답 라벨과 모델 예측을 표시하지 않아 검토를 유도하지 않는다.\n','각 쌍의 판정에 감성 유지 / 감성 변화 / 판단 불가 중 하나를 적고, 이유를 짧게 적는다. 문장이 같으면 동일 표시를 참고하되 실제 내용을 확인한다. 불쾌한 표현이 원자료에 포함될 수 있다.\n']
    review_records=[]
    for j,rid in enumerate(review_ids,1):
        review.append(f'## 리뷰 {j:02d} (ID {rid})\n\n원문: '+original[rid].replace('\n',' ')+'\n')
        for rate in RATES[1:]:
            same=original[rid]==variants[rid,rate]
            review.extend([f'### 삭제 확률 {rate:.2f} / 원문과 동일: {"예" if same else "아니오"}',variants[rid,rate], '\n판정: 미검토\n\n이유: \n'])
            review_records.append({'id':rid,'rate':rate,'unchanged':same,'judgment':None,'reason':None})
    (a.out/'학생_감성검토_150쌍.md').write_text('\n'.join(review))
    (a.out/'manual_review_template.json').write_text(json.dumps(review_records,ensure_ascii=False,indent=2))
    # Random sample independent of correctness, then attach predictions for error analysis.
    sample=random.Random(1105).sample(ids,20);examples=[]
    for rid in sample:
        i=pos[rid];examples.append({'id':rid,'original':original[rid],'perturbed_p1':variants[rid,1.0],'gold':int(labels[i]),'seed42_predictions_p1':{f'{f}_{"aug" if aug else "base"}':int(pred[0,k,aug,3,i]) for k,f in enumerate(FEATURES) for aug in [0,1]}})
    (a.out/'random_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
