"""CPU-only NSMC spacing-deletion pilot. Default uses only official training data."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()


def key(text):
    # Also prevent differently spaced copies from crossing the data split.
    return re.sub(r'\s+', '', text)


def corrupt(text, rate, seed, row_id):
    rng = random.Random(f'{seed}:{row_id}')
    return ''.join(c for c in text if c != ' ' or rng.random() >= rate)


def load(path):
    groups = {}
    with path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            text = normalize(row['document'])
            if not text:
                continue
            entry = (row['id'], text, int(row['label']))
            groups.setdefault(key(text), []).append(entry)
    # Drop contradictory-label duplicates, keep one of same-label duplicates.
    return [v[0] for v in groups.values() if len({x[2] for x in v}) == 1]


def subset(rows, count, seed):
    if count >= len(rows):
        return rows
    return train_test_split(rows, train_size=count, stratify=[r[2] for r in rows], random_state=seed)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path('data'))
    ap.add_argument('--out', type=Path, default=Path('results/pilot'))
    ap.add_argument('--train-size', type=int, default=2000)
    ap.add_argument('--eval-size', type=int, default=500)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--max-features', type=int, default=30000)
    ap.add_argument('--final-test', action='store_true', help='Use only AFTER design is frozen.')
    args = ap.parse_args()
    if args.train_size < 10 or args.eval_size < 10:
        ap.error('Sample sizes must be at least 10.')
    args.out.mkdir(parents=True, exist_ok=False)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = args.data_dir / 'source.json'
    if source.exists():
        ref = json.loads(source.read_text())['commit']
    else:
        req = urllib.request.Request('https://api.github.com/repos/e9t/nsmc/commits/master', headers={'User-Agent': 'NLP-course-pilot'})
        with urllib.request.urlopen(req, timeout=60) as response:
            ref = json.load(response)['sha']
        source.write_text(json.dumps({'repository': 'https://github.com/e9t/nsmc', 'commit': ref}, indent=2))
    names = ['ratings_train.txt'] + (['ratings_test.txt'] if args.final_test else [])
    for name in names:
        target = args.data_dir / name
        if not target.exists():
            url = f'https://raw.githubusercontent.com/e9t/nsmc/{ref}/{name}'
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            target.write_bytes(content)
    rows = load(args.data_dir / 'ratings_train.txt')
    training_pool, validation_pool = train_test_split(rows, test_size=0.2, stratify=[r[2] for r in rows], random_state=2026)
    if args.final_test:
        known = {key(r[1]) for r in rows}
        evaluation_pool = [r for r in load(args.data_dir / 'ratings_test.txt') if key(r[1]) not in known]
    else:
        evaluation_pool = validation_pool
    evaluation = subset(evaluation_pool, args.eval_size, 2026)
    y_eval = [r[2] for r in evaluation]
    metrics, split_ids = [], {'evaluation': [r[0] for r in evaluation], 'training': {}}
    predictions = (args.out / 'predictions.jsonl').open('w', encoding='utf-8')
    for seed in args.seeds:
        train = subset(training_pool, args.train_size, seed)
        assert not ({key(r[1]) for r in train} & {key(r[1]) for r in evaluation})
        split_ids['training'][str(seed)] = [r[0] for r in train]
        for feature in ['word', 'char']:
            for augmented in [False, True]:
                texts = [r[1] for r in train]
                labels = [r[2] for r in train]
                if augmented:
                    texts += [corrupt(r[1], 0.5, seed + 1000, r[0]) for r in train]
                    labels *= 2
                if feature == 'word':
                    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2, max_features=args.max_features, sublinear_tf=True)
                else:
                    # 'char', NOT 'char_wb': retain cross-boundary character features.
                    vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, ngram_range=(2, 5), min_df=2, max_features=args.max_features, sublinear_tf=True)
                model = make_pipeline(vectorizer, LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=seed))
                fit_start = time.perf_counter()
                # Average loss has equal original-example weight across conditions.
                weights = np.full(len(labels), 0.5 if augmented else 1.0)
                model.fit(texts, labels, logisticregression__sample_weight=weights)
                fit_seconds = time.perf_counter() - fit_start
                clean_f1 = None
                for rate in [0.0, 0.25, 0.5, 1.0]:
                    noisy = [corrupt(r[1], rate, 9000, r[0]) for r in evaluation]
                    pred = model.predict(noisy)
                    f1 = f1_score(y_eval, pred, average='macro', zero_division=0)
                    if rate == 0:
                        clean_f1 = f1
                    record = dict(seed=seed, feature=feature, augmented=augmented, deletion_probability=rate, accuracy=accuracy_score(y_eval, pred), macro_f1=f1, f1_drop=clean_f1-f1, changed_fraction=sum(x != r[1] for x,r in zip(noisy,evaluation))/len(evaluation), fit_seconds=fit_seconds, vocabulary_size=len(vectorizer.vocabulary_), n_train_original=len(train), n_eval=len(evaluation))
                    metrics.append(record)
                    print(json.dumps(record), flush=True)
                    for row, text, prediction in zip(evaluation, noisy, pred):
                        predictions.write(json.dumps(dict(seed=seed, feature=feature, augmented=augmented, deletion_probability=rate, id=row[0], label=row[2], prediction=int(prediction), original=row[1], perturbed=text), ensure_ascii=False)+'\n')
    predictions.close()
    metadata = {'status': 'completed', 'started_scope': 'final_test' if args.final_test else 'development_pilot', 'utc_completed': datetime.now(timezone.utc).isoformat(), 'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, 'command': sys.argv, 'python': sys.version, 'platform': platform.platform(), 'versions': {p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn']}, 'dataset_commit': ref, 'data_sha256': {n:sha(args.data_dir/n) for n in names}, 'script_sha256': sha(Path(__file__)), 'total_seconds': time.perf_counter()-start, 'deduplicated_official_train': len(rows), 'split_seed': 2026, 'noise_seed': 9000, 'note': 'Original NSMC text may already contain noise. Word means whitespace-delimited eojeol, not morphological word.'}
    (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (args.out/'split_ids.json').write_text(json.dumps(split_ids,indent=2))


if __name__ == '__main__':
    main()
