# HW2 NSMC spacing experiments

This package contains the development pipeline and the frozen final-test evaluation. The author chose a0.01 original-performance tolerance after examining development results. The final-test rule is a strict95% paired-bootstrap lower bound greater than−0.01. Settings and rule were frozen before downloading or evaluating the official test file. This is not preregistration.

## Environment

Python3.12.14, macOS26.6.2 arm64. Install the exact versions in requirements.txt in a fresh virtual environment. The November24 reproduction used a newly created virtual environment and freshly installed packages on the same computer. This is not verification on a second physical machine. Internet is needed for installation and for downloading missing data. The data/source.json file pins the NSMC commit.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Final main score table and confidence intervals

Run from this directory. Each results directory must be new.

```sh
python run_experiment.py --data-dir data --train-size 10000 --eval-size 2000 --seeds 42 43 44 --final-test --out results/final_test
python analyze_development.py results/final_test --out results/final_analysis --final-test --bootstrap 1000 --bootstrap-seed 11042026
python verify_reproduction.py reference-final results/final_test
```

The first command produces48 metric rows and96,000 predictions. The second verifies saved scores against predictions and creates RESULTS.md and analysis.json. The third compares scores, split IDs, data hashes, and code hash against the included reference. Runtime is excluded from comparison. Predictions include original review texts and are regenerated locally, not included in the archive.

Final table: seed-average original and deletion-condition macro-F1, mean of the three perturbed macro-F1 scores, and original-minus-perturbed drop. Sample standard deviations appear in analysis.json. The five comparison intervals use1000 ordinary review-ID bootstrap resamples with seed11042026. Every model/rate/seed shares each review resample. These percentile95% intervals are conditional on trained models; no multiple-comparison correction is applied.

## Development results and feature coverage

```sh
python run_experiment.py --data-dir data --train-size 10000 --eval-size 2000 --seeds 42 43 44 --out results/development
python analyze_development.py results/development --out results/development_analysis --bootstrap 1000 --bootstrap-seed 11042026
python diagnose_features.py --data-dir data --results results/development --out results/development_analysis/feature_coverage.json
```

The feature diagnostic is intended for development data. It writes feature_coverage.json and feature_coverage.md. It fits the same TF-IDF configurations to the recorded training IDs and measures zero-vector proportions. It does not refit classifiers or establish every cause of an error.

## Frozen settings

Normalize whitespace; remove empty reviews; deduplicate whitespace-stripped content; discard contradictory-label groups. Split the official training data into stratified80:20 pools with seed2026. Select10000 training reviews for each seed42/43/44 and2000 fixed development reviews with seed2026. Final evaluation selects2000 reviews from the official test data after filtering overlaps with the cleaned official training pool. All four models share the training originals within a seed. Evaluation noise seed9000 is fixed across models and seeds.

Eojeol features use whitespace-delimited1–2-grams. Character features use analyzer='char',2–5-grams, retaining spaces and cross-boundary features. Both use min_df=2,max_features=30000,sublinear_tf=True,lowercase=False. Logistic regression uses C=1,solver='liblinear',max_iter=1000. Augmentation adds one variant per original using independent per-space deletion probability0.5. Original and augmented example each receive classifier sample weight0.5. TF-IDF itself uses the corresponding full corpus without those weights.

H1: the lower95% interval bound of eojeol mean drop minus character mean drop must exceed0 to support the directional hypothesis. H2, separately by representation: the perturbed-score gain lower bound must exceed0 AND original-score change lower bound must exceed−0.01. Intervals crossing a boundary are inconclusive. The0.005 sensitivity criterion is secondary and must not replace the chosen0.01 rule. See frozen_design_20261124.json for the pre-test record.

## Human inspection and random cases

The development script generates150 review pairs from50 source reviews, sampling seed1104 independently of correctness. The author supplied the completed judgments:130 changed pairs judged preserved and20 unchanged pairs marked separately. See manual-review/manual_review_annotations.json and manual_review_summary.json. The generator produces an unfilled template; it does not recreate human judgments. The original label can remain ambiguous, so preservation does not verify the binary gold label. One rater and repeated variants of50 sources limit inference.

The development script also generates20 cases at seed1105 before examining correctness. These are development examples, not final-test anecdotes. Cases include successes and failures. Do not report20 cases as all independent evidence about population error rates.

## Reproducibility and disclosure

Codex executed the experiments, analysis and reproduction check. The author supplied the motivation, Korean drafts, interpretation of development results,150 human judgments, and the tolerance decision. Do not describe AI-executed experiments as unaided author execution. The author identified the HW2 model as Astra 6.2. Earlier HW1 work used GPT-6 Astra, as reported by the author.

The archive excludes raw dataset files, prediction texts, model checkpoints and installed environments. Source: https://github.com/e9t/nsmc . No clear redistribution license is asserted. A shareable code URL and independent-machine execution remain unverified; local ZIP preparation alone does not fulfill those requirements. No file is uploaded by these commands.


## Frozen decisions and prediction-count analysis

```sh
python summarize_final.py results/final_test results/final_analysis frozen_design_20261124.json --out results/final_audit
```

This writes the strict0.01/0.005 decisions and the class-prediction counts used in the paper. It does not fit or tune any model.
