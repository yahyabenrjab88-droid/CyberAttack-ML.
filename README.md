# CICIDS2017 Network Intrusion Detection Using Machine Learning

**A reproducible machine-learning project for multi-class network intrusion detection, with a documented experimental journey from baseline modeling to final evaluation.**

[!\[Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[!\[Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-0A7EA4)](https://www.unb.ca/cic/datasets/ids-2017.html)
[!\[Model](https://img.shields.io/badge/Model-HistGradientBoosting-245A8D)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
[!\[Task](https://img.shields.io/badge/Task-Multiclass%20Classification-168A72)](#problem-definition)
[!\[License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

\---

## Contents

* [Project Overview](#project-overview)
* [Results at a Glance](#results-at-a-glance)
* [Problem Definition](#problem-definition)
* [Dataset and Target Labels](#dataset-and-target-labels)
* [Methodology](#methodology)
* [Data Quality and Preprocessing](#data-quality-and-preprocessing)
* [Experimental Journey](#experimental-journey)

  * [Strategy 1 — Baseline](#strategy-1--baseline)
  * [Strategy 2 — BF/XSS Specialist](#strategy-2--bfxss-specialist)
  * [Strategy 3 — Gated Calibrated Specialist](#strategy-3--gated-calibrated-specialist)
  * [Strategy 4 — Hard-Example Specialist](#strategy-4--hard-example-specialist)
  * [Strategy 5 — Out-of-Fold Stacked Meta-Learner](#strategy-5--out-of-fold-stacked-meta-learner)
  * [Strategy 6 — SMOTE-Tomek Specialist](#strategy-6--smote-tomek-specialist)
  * [Strategy 7 — Out-of-Fold SMOTE-Tomek](#strategy-7--out-of-fold-smote-tomek)
  * [Strategy 8 — Merged Web Attack Model](#strategy-8--merged-web-attack-model)
  * [Strategy 8b — Hulk/Benign Threshold Investigation](#strategy-8b--hulkbenign-threshold-investigation)
* [Final Test Evaluation](#final-test-evaluation)
* [Repository Structure](#repository-structure)
* [Environment Setup](#environment-setup)
* [Reproducing the Workflow](#reproducing-the-workflow)
* [Interpretation and Limitations](#interpretation-and-limitations)
* [Future Work](#future-work)
* [Dataset Attribution and License](#dataset-attribution-and-license)

\---

## Project Overview

This repository documents an end-to-end machine-learning workflow for multi-class network intrusion detection using the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html), published by the Canadian Institute for Cybersecurity at the University of New Brunswick.

The workflow includes data-quality investigation, cleaning, split construction, leakage-conscious preprocessing, model development, specialist experiments, error analysis, and a final held-out test evaluation. The repository preserves the experimental journey rather than presenting only the selected model, making it possible to inspect the decisions and trade-offs that shaped the final approach.

### Objectives

* Build a consistent and auditable pipeline for network-flow classification.
* Investigate data-quality issues, including missing values, infinities, duplicates, and class distribution.
* Prevent leakage between training, validation, and test data during preprocessing and model development.
* Compare multiple modeling approaches and retain evidence from approaches that were not selected.
* Evaluate performance with aggregate metrics and class-level results.
* Explain the final target-label design, its benefits, and its limitations.
* Keep model artifacts, feature ordering, label ordering, and evaluation outputs organized.

### Project scope

This is an **offline, dataset-based classification project**. It does not currently constitute a live packet-capture sensor, a real-time monitoring service, or a production intrusion-prevention system. The reported metrics describe performance on the project's prepared CICIDS2017 data and should not be interpreted as a guarantee of performance on live or unseen network traffic.

\---

## Results at a Glance

The selected Strategy 8 model is a HistGradientBoosting classifier trained on a 13-class target formulation in which the three Web Attack labels are merged into one broader class.

### Reported Strategy 8 performance

|Metric|Validation|Held-out test|
|-|-:|-:|
|Macro-F1|96.5579%|**96.6124%**|
|Accuracy|99.16%|**99.54%**|
|Weighted-F1|99.16%|**99.55%**|

These figures are the results recorded for the project; they have not been independently reproduced as part of preparing this README. Validation and test metrics come from different splits and must not be treated as measurements on the same examples.

Macro-F1 is highlighted because accuracy and weighted-F1 can be dominated by frequent classes in an imbalanced dataset. Aggregate scores should be considered alongside per-class precision, recall, F1-score, support, and the confusion matrix.

> \\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*Evaluation discipline:\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\* model selection, feature decisions, label-mapping decisions, and threshold tuning belong in the development/validation process. The held-out test split should be reserved for final evaluation rather than repeatedly used to guide experiments.

\---

## Problem Definition

The task is supervised multi-class classification of labeled network-flow records. It is broader than a binary benign-versus-malicious decision: the classifier predicts one of the target classes defined by the project's label mapping.

The Strategy 8 formulation contains **13 classes**, including a merged `Web Attack` class. This label design is a deliberate change to the classification target and affects what the model can report.

### Feature contract

The processed train and validation datasets each contain 70 columns:

* **68 numeric model features**
* `source\\\\\\\\\\\\\\\_file`: source/provenance metadata
* `Label`: target class

The model must use the same 68 numeric features, in the same order, during training and inference. The `source\\\\\\\\\\\\\\\_file` column is metadata and must never be used as a model feature. The `Label` column is the target and must also be excluded from the feature matrix.

The saved feature-column artifact is intended to support a consistent feature contract across model training and prediction.

\---

## Dataset and Target Labels

The project uses the CICIDS2017 network intrusion detection dataset. It contains labeled flow records representing benign traffic and multiple attack categories.

* Official dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
* Publisher: Canadian Institute for Cybersecurity, University of New Brunswick

Raw and processed dataset files are not included in this repository because of size and redistribution considerations. Obtain the data from the official source, review the applicable terms, and place the required source files in the locations expected by the pipeline scripts.

### Why merge the Web Attack labels?

Strategy 8 maps the following labels into one broader `Web Attack` class:

* Web Attack – Brute Force
* Web Attack – XSS
* Web Attack – Sql Injection

Earlier experiments investigated the difficulty of separating Brute Force and XSS and the limits of adding increasingly specialized decision logic. Strategy 8 changes the target-label design by combining the three Web Attack subclasses into one category.

This is a **modeling trade-off**, not a claim that the underlying attacks are identical. The merged class makes the task less granular: the model can identify the broader Web Attack family, but it does not distinguish these three subtypes from one another.

The exact normalization and mapping of raw labels should be verified in the data-cleaning and final-strategy scripts, since raw dataset copies can differ in label spelling or punctuation.

\---

## Methodology

The workflow is organized into the following stages:

```text
CICIDS2017 source files
        |
        v
Data-quality investigation and cleaning
        |
        v
Train / validation / test split
        70% / 15% / 15%
        |
        v
Preprocessing fitted on training data only
  - positive/negative infinity -> missing values
  - training-only median imputation
  - selected missingness indicators
        |
        v
Baseline and historical experiments
        |
        v
Model comparison and error analysis
        |
        v
Strategy 8: 13-class HistGradientBoosting
        |
        +----> Validation analysis / Strategy 8b investigation
        |
        v
Held-out test evaluation
```

The percentages describe the intended split proportions. Use the split implementation and saved manifests to verify the exact grouping and separation guarantees; the proportions alone do not establish how records or source files were grouped.

\---

## Data Quality and Preprocessing

Preprocessing choices are important to the validity of the reported results. The project avoids blindly replacing missing values with zero and fits median-imputation statistics on training data only.

### Infinite and missing values

* Positive and negative infinity are converted to missing values.
* Missing numeric values are imputed using medians fitted on the training split.
* The fitted imputation statistics are then applied to validation and test data.
* Missing values are **not** indiscriminately replaced with zero.

Fitting imputation statistics on the entire dataset would allow information from validation or test data to influence preprocessing learned during training. Fitting on the training split only avoids this particular leakage pathway.

### Missingness indicators

The selected preprocessing configuration includes missingness indicators for:

* `Flow Bytes/s`
* `Flow IAT Min`
* `Fwd IAT Min`

`Flow Packets/s\\\\\\\\\\\\\\\_missing` is omitted because it was found to duplicate `Flow Bytes/s\\\\\\\\\\\\\\\_missing` in the audited data. Keeping a duplicate indicator would add no distinct signal.

### Split and feature configuration

|Item|Configuration|
|-|-|
|Training split|70%|
|Validation split|15%|
|Held-out test split|15%|
|Numeric model features|68|
|Target column|`Label`|
|Metadata excluded from features|`source\\\\\\\\\\\\\\\_file`|
|Median-imputation fit|Training split only|

Before training or inference, verify that feature construction, column order, label mapping, and preprocessing artifacts match the configuration used for the selected model.

\---

# Experimental Journey

The following experiments document the progression from a baseline classifier to specialist approaches and finally a revised target-label formulation. Historical experiments remain valuable even when they are not selected: they show which hypotheses were tested, what trade-offs appeared, and why an approach should or should not be carried forward.

## Summary of Strategies 1–8

|Strategy|Approach|Reported validation Macro-F1|BF/XSS mutual confusion|Status / interpretation|
|-|-|-:|-:|-|
|1|HistGradientBoosting baseline, 13 classes|85.8601%|108|Reference baseline|
|2|BF/XSS specialist|86.3277%|106|Small aggregate gain; limited change in targeted confusion|
|3|Gated calibrated specialist, threshold 0.55|84.77%|88|Lower targeted confusion but lower aggregate Macro-F1|
|4|Hard-example specialist|See saved experiment report|See saved experiment report|Historical experiment; verify exact results|
|5|Out-of-fold stacked meta-learner|See saved experiment report|See saved experiment report|Historical experiment; verify exact results|
|6|SMOTE-Tomek specialist|See saved experiment report|See saved experiment report|Historical experiment; verify exact results|
|7|Out-of-fold SMOTE-Tomek|See saved experiment report|See saved experiment report|Historical experiment; verify exact results|
|8|HistGradientBoosting with merged Web Attack labels|96.5579%|Not directly comparable after label merge|Selected final formulation|

**About the table:** the Strategy 1–3 figures are the recorded values available for this README. Exact Strategy 4–7 metrics should be transcribed from the saved result artifacts before publication; they are intentionally not guessed. BF/XSS confusion counts also become non-comparable after the relevant target labels are merged.

### Strategy 1 — Baseline

**Objective:** establish a reference point before introducing specialized decision logic.

**Approach:** train a HistGradientBoosting classifier on the original 13-class formulation used in the baseline experiment.

**Reported validation results:**

* Macro-F1: **85.8601%**
* BF/XSS mutual confusion: **108**

**Interpretation:** the baseline provided a consistent comparison point and made the BF/XSS distinction visible as a specific error-analysis problem. A model can perform strongly overall while still struggling to separate a particular pair of classes.

**Engineering takeaway:** establish a reproducible baseline and inspect class-level errors before adding complexity. Targeted modeling decisions should be motivated by measured failure patterns.

**Implementation:** `src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_baseline.py`

### Strategy 2 — BF/XSS Specialist

**Objective:** investigate whether a specialist model can reduce confusion between Web Attack Brute Force (BF) and XSS.

**Approach:** train a specialist for the BF/XSS distinction and integrate it into the primary classification workflow. The precise routing conditions and combination logic are defined by the experiment script and its saved outputs.

**Reported validation results:**

* Macro-F1: **86.3277%**
* BF/XSS mutual confusion: **106**

**Outcome:** Macro-F1 increased slightly compared with the baseline, while BF/XSS mutual confusion changed from 108 to 106. The targeted change was limited, so the results did not establish a compelling overall improvement by themselves.

**Engineering takeaway:** specialist models should be assessed at two levels: whether they improve the targeted error pattern and whether the intervention improves the complete multi-class system. A narrow gain may not justify extra model complexity.

**Implementation:** `src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_specialist\\\\\\\\\\\\\\\_cascade.py`

### Strategy 3 — Gated Calibrated Specialist

**Objective:** investigate whether a specialist should intervene only when a gate indicates that its prediction is appropriate.

**Approach:** use a calibrated, gated specialist with a reported threshold of **0.55**. The exact calibration method, gate input, and routing behavior should be confirmed from the implementation.

**Reported validation results:**

* Macro-F1: **84.77%**
* BF/XSS mutual confusion: **88**

**Outcome:** the targeted BF/XSS mutual-confusion count decreased, but aggregate Macro-F1 fell below the reported results for Strategies 1 and 2.

**Engineering takeaway:** a threshold can reduce a targeted error while introducing errors elsewhere. A gate must be evaluated against the full validation metric, per-class metrics, and error distribution—not only the specific class pair it was designed to address.

**Implementation:** `src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_gated\\\\\\\\\\\\\\\_calibrated\\\\\\\\\\\\\\\_cascade.py`

### Strategy 4 — Hard-Example Specialist

**Objective:** investigate whether explicitly focusing on difficult examples could improve the specialist's decision boundary.

**Approach:** use a hard-example specialist/cascade approach. The exact definition of a hard example, selection procedure, specialist training configuration, and routing behavior should be taken from the script and saved experiment report.

**What to inspect in the experiment artifacts:**

* How were hard examples identified?
* Was hard-example selection based only on permitted training data?
* Which examples were routed to the specialist?
* Did targeted corrections introduce additional errors in other classes?
* What were the complete validation metrics and confusion counts?

**Interpretation:** this experiment is retained as part of the project's technical history. Its result should be described using the actual saved metrics rather than inferred from the strategy name.

**Implementation:** `src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_hard\\\\\\\\\\\\\\\_example\\\\\\\\\\\\\\\_specialist.py`

### Strategy 5 — Out-of-Fold Stacked Meta-Learner

**Objective:** test whether a meta-model could learn when and how to combine predictions from the primary classifier and specialist.

**Approach:** generate out-of-fold (OOF) predictions and train a logistic-regression meta-model. OOF predictions are used so that the meta-model can learn from base-model outputs generated without training the base model on the corresponding row.

**Technical considerations:**

* OOF predictions must respect the intended fold boundaries.
* The meta-model should not train on optimistic in-sample base-model predictions where OOF predictions are required.
* Validation and test labels must not leak into the construction of meta-training features.
* The complete multi-class result should be compared with simpler routing approaches.

**What to inspect in the artifacts:** fold configuration, meta-feature construction, base-model outputs, class ordering, split discipline, and the saved validation metrics.

**Implementation:** `src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_oof\\\\\\\\\\\\\\\_stacked\\\\\\\\\\\\\\\_metalearner.py`

### Strategy 6 — SMOTE-Tomek Specialist

**Objective:** investigate whether resampling could improve the specialist's handling of difficult or imbalanced class examples.

**Approach:** use a SMOTE-Tomek resampling approach. SMOTE creates synthetic minority-class examples; Tomek-link cleaning removes selected close pairs from opposing classes. Resampling changes the training distribution and may alter local decision boundaries.

**Leakage consideration:** resampling must be restricted to the relevant training portion. Validation and test data must remain untouched by synthetic sample generation or resampling-based selection.

**What to inspect in the artifacts:**

* The exact training subset on which resampling was performed.
* The order of preprocessing and resampling.
* Precision/recall trade-offs for the targeted classes.
* Whether the complete validation metric improved, rather than only the specialist subset.

**Interpretation:** resampling is an experiment, not an automatic solution to class imbalance. Its value depends on generalization and the trade-off between false positives and false negatives.

**Implementation:** `src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_smote\\\\\\\\\\\\\\\_tomek\\\\\\\\\\\\\\\_specialist.py`

### Strategy 7 — Out-of-Fold SMOTE-Tomek

**Objective:** investigate a combination of OOF stacking and resampling-aware specialist training.

**Approach:** use an out-of-fold SMOTE-Tomek workflow. To avoid leakage, resampling should be performed independently within each fold's training partition rather than before fold construction in a way that allows synthetic or related information to cross boundaries.

**What to inspect in the artifacts:**

* Whether resampling occurred independently inside each training fold.
* Whether every OOF prediction was generated without training the corresponding base model on that row.
* Which predictions or derived features were supplied to the meta-model.
* Whether the approach improved full validation performance relative to the standalone specialist and other stacking strategy.

**Interpretation:** combining resampling and stacking increases pipeline complexity. It should be retained only when the saved evaluation evidence justifies that complexity.

**Implementation:** `src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_oof\\\\\\\\\\\\\\\_smote\\\\\\\\\\\\\\\_tomek.py`

### Strategy 8 — Merged Web Attack Model

**Objective:** address persistent overlap by reconsidering the target-label design instead of continuing to add increasingly specialized decision logic for Web Attack subclasses.

**Approach:** train a HistGradientBoosting classifier on the 68 numeric features while merging Web Attack – Brute Force, Web Attack – XSS, and Web Attack – Sql Injection into one broader `Web Attack` class. The resulting task has 13 classes.

**Reported validation metrics:**

* Macro-F1: **96.5579%**
* Accuracy: **99.16%**
* Weighted-F1: **99.16%**

**Reported held-out test metrics:**

* Macro-F1: **96.6124%**
* Accuracy: **99.54%**
* Weighted-F1: **99.55%**

**Why this approach was selected:** the strategy changes the target taxonomy to a broader family-level prediction task. It addresses the practical limitation of separating highly overlapping Web Attack subclasses in the earlier experiments and avoids treating added specialist complexity as inherently beneficial. It also retains the defined 68-feature input contract.

**Trade-off:** the final classifier no longer distinguishes Brute Force, XSS, and SQL Injection within the merged class. This must be disclosed when interpreting results or describing the model's capabilities.

**Saved model artifacts:**

* `models/strategy8\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged/hgb\\\\\\\\\\\\\\\_13class\\\\\\\\\\\\\\\_model.joblib`
* `models/strategy8\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged/feature\\\\\\\\\\\\\\\_columns.joblib`
* `models/strategy8\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged/class\\\\\\\\\\\\\\\_order.joblib`

**Implementation:** `src/final\\\\\\\\\\\\\\\_strategy/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged.py`

### Strategy 8b — Hulk/Benign Threshold Investigation

Strategy 8b investigates a decision-threshold adjustment associated with the DoS Hulk/Benign error pattern. The reported validation Macro-F1 for the investigated configuration was approximately **96.6581%**.

This result belongs to a validation threshold experiment. It must not be substituted for the Strategy 8 held-out test score or described as a separate final test result.

Keep three concepts distinct:

1. **Selected Strategy 8 model:** the model formulation and saved artifacts chosen for the final strategy.
2. **Strategy 8b validation investigation:** a threshold experiment used to study a specific error pattern.
3. **Held-out test evaluation:** the final reported test metrics for the selected evaluation configuration.

Thresholds should be chosen using development/validation data. Repeatedly tuning decisions against the held-out test split would weaken the independence of the final evaluation.

**Implementation:** `src/final\\\\\\\\\\\\\\\_strategy/strategy8b\\\\\\\\\\\\\\\_hulk\\\\\\\\\\\\\\\_benign\\\\\\\\\\\\\\\_thresholds.py`

\---

## Final Test Evaluation

The reported held-out test results for Strategy 8 are:

|Metric|Result|
|-|-:|
|Macro-F1|96.6124%|
|Accuracy|99.54%|
|Weighted-F1|99.55%|

The following class-level figures are the values available for this README. They should be checked against the saved classification report if the artifacts have changed.

|Class|Precision|Recall|F1-score / support|
|-|-:|-:|-:|
|Bot|74.62%|98.64%|F1: 84.96%|
|Web Attack (merged)|93.33%|98.47%|F1: 95.83%|
|Heartbleed|—|—|Support: 2|
|Infiltration|—|—|Support: 5|

Heartbleed and Infiltration have very small test support. Their metrics are therefore highly uncertain and should not be treated as stable estimates of performance on future data.

### How to interpret the metrics

* **Macro-F1:** computes F1 for each class and averages the class scores equally. It is useful when minority classes should matter even if they have few examples.
* **Accuracy:** the proportion of all evaluated examples classified correctly.
* **Weighted-F1:** averages class-level F1 scores weighted by each class's support. Frequent classes have more influence.
* **Precision:** among examples predicted as a class, the proportion that truly belongs to that class.
* **Recall:** among examples that truly belong to a class, the proportion identified by the model.
* **Support:** the number of true examples of a class in the evaluated split.

No single metric captures every operational cost. In an intrusion-detection setting, false positives can create alert fatigue, while false negatives can allow malicious activity to go undetected. A deployment decision would require evaluation aligned with the intended environment and response workflow.

\---

## Repository Structure

The tree below is a navigational overview based on the project structure documented for this repository. If paths have changed in the current checkout, the actual repository tree is authoritative.

```text
CyberAttack-ML/
├── data/
│   ├── raw/                         # Local source data; not committed
│   └── processed/                   # Generated data; not committed
├── models/
│   ├── historical/
│   └── strategy8\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged/
│       ├── hgb\\\\\\\\\\\\\\\_13class\\\\\\\\\\\\\\\_model.joblib
│       ├── feature\\\\\\\\\\\\\\\_columns.joblib
│       └── class\\\\\\\\\\\\\\\_order.joblib
├── reports/
│   ├── data\\\\\\\\\\\\\\\_quality/
│   ├── duplicates/
│   ├── investigations/
│   └── preprocessing\\\\\\\\\\\\\\\_split/
├── results/
│   ├── baseline/
│   ├── strategy4\\\\\\\\\\\\\\\_hardweighted\\\\\\\\\\\\\\\_specialist\\\\\\\\\\\\\\\_cascade/
│   ├── strategy5\\\\\\\\\\\\\\\_oof\\\\\\\\\\\\\\\_stacked\\\\\\\\\\\\\\\_metalearner/
│   ├── strategy6\\\\\\\\\\\\\\\_smote\\\\\\\\\\\\\\\_tomek/
│   ├── strategy7\\\\\\\\\\\\\\\_oof\\\\\\\\\\\\\\\_smote\\\\\\\\\\\\\\\_tomek/
│   ├── strategy8\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged/
│   └── final\\\\\\\\\\\\\\\_test\\\\\\\\\\\\\\\_evaluation\\\\\\\\\\\\\\\_threshold025/
├── src/
│   ├── data\\\\\\\\\\\\\\\_pipeline/
│   ├── final\\\\\\\\\\\\\\\_strategy/
│   ├── historical\\\\\\\\\\\\\\\_experiments/
│   ├── investigations/
│   └── validation/
├── requirements.txt
├── requirements\\\\\\\\\\\\\\\_before\\\\\\\\\\\\\\\_strategy6.txt
├── src\\\\\\\\\\\\\\\_inventory.txt
├── .gitignore
├── LICENSE
└── README.md
```

### Important entry points

|Purpose|Script|
|-|-|
|Clean source data|`src/data\\\\\\\\\\\\\\\_pipeline/clean.py`|
|Create data splits|`src/data\\\\\\\\\\\\\\\_pipeline/split.py`|
|Preprocess split data|`src/data\\\\\\\\\\\\\\\_pipeline/preprocess.py`|
|Audit cleaned data|`src/validation/audit\\\\\\\\\\\\\\\_cleaned.py`|
|Audit split integrity|`src/validation/audit\\\\\\\\\\\\\\\_splits.py`|
|Audit preprocessed features|`src/validation/audit\\\\\\\\\\\\\\\_preprocessed.py`|
|Train baseline|`src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_baseline.py`|
|Train Strategy 8|`src/final\\\\\\\\\\\\\\\_strategy/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged.py`|
|Analyze Strategy 8 errors|`src/final\\\\\\\\\\\\\\\_strategy/analyze\\\\\\\\\\\\\\\_strategy8\\\\\\\\\\\\\\\_errors.py`|
|Investigate Strategy 8b thresholds|`src/final\\\\\\\\\\\\\\\_strategy/strategy8b\\\\\\\\\\\\\\\_hulk\\\\\\\\\\\\\\\_benign\\\\\\\\\\\\\\\_thresholds.py`|
|Evaluate Strategy 8 on test data|`src/final\\\\\\\\\\\\\\\_strategy/evaluate\\\\\\\\\\\\\\\_strategy8\\\\\\\\\\\\\\\_test.py`|

\---

## Environment Setup

The following example uses Windows PowerShell and a local Python virtual environment.

```powershell
cd C:\\\\\\\\\\\\\\\\CyberAttack-ML

py -m venv .venv
.\\\\\\\\\\\\\\\\.venv\\\\\\\\\\\\\\\\Scripts\\\\\\\\\\\\\\\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation because of the current execution policy, use an approved local environment configuration rather than changing system-wide policy unnecessarily.

### Dataset placement

Download CICIDS2017 from the [official dataset page](https://www.unb.ca/cic/datasets/ids-2017.html). Place the required source files where `src/data\\\\\\\\\\\\\\\_pipeline/clean.py` expects them, and inspect that script's configuration before running it. The dataset itself is not included in this repository.

\---

## Reproducing the Workflow

Run the pipeline in stages. Before execution, inspect each script's current arguments, input paths, random seeds, output paths, and overwrite behavior. Some experiments may be computationally expensive. The commands below document the intended workflow; they have not been executed as part of preparing this README.

### 1\. Clean, split, and preprocess the data

```powershell
python src/data\\\\\\\\\\\\\\\_pipeline/clean.py
python src/data\\\\\\\\\\\\\\\_pipeline/split.py
python src/data\\\\\\\\\\\\\\\_pipeline/preprocess.py
```

### 2\. Run data-quality and preprocessing audits

```powershell
python src/validation/audit\\\\\\\\\\\\\\\_cleaned.py
python src/validation/audit\\\\\\\\\\\\\\\_splits.py
python src/validation/audit\\\\\\\\\\\\\\\_preprocessed.py
```

### 3\. Reproduce the baseline and historical experiments

```powershell
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_baseline.py
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_specialist\\\\\\\\\\\\\\\_cascade.py
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_gated\\\\\\\\\\\\\\\_calibrated\\\\\\\\\\\\\\\_cascade.py
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_hard\\\\\\\\\\\\\\\_example\\\\\\\\\\\\\\\_specialist.py
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_hardweighted\\\\\\\\\\\\\\\_specialist\\\\\\\\\\\\\\\_cascade.py
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_oof\\\\\\\\\\\\\\\_stacked\\\\\\\\\\\\\\\_metalearner.py
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_smote\\\\\\\\\\\\\\\_tomek\\\\\\\\\\\\\\\_specialist.py
python src/historical\\\\\\\\\\\\\\\_experiments/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_bfxss\\\\\\\\\\\\\\\_oof\\\\\\\\\\\\\\\_smote\\\\\\\\\\\\\\\_tomek.py
```

### 4\. Train, analyze, and evaluate Strategy 8

```powershell
python src/final\\\\\\\\\\\\\\\_strategy/train\\\\\\\\\\\\\\\_hgb\\\\\\\\\\\\\\\_web\\\\\\\\\\\\\\\_attack\\\\\\\\\\\\\\\_merged.py
python src/final\\\\\\\\\\\\\\\_strategy/analyze\\\\\\\\\\\\\\\_strategy8\\\\\\\\\\\\\\\_errors.py
python src/final\\\\\\\\\\\\\\\_strategy/strategy8b\\\\\\\\\\\\\\\_hulk\\\\\\\\\\\\\\\_benign\\\\\\\\\\\\\\\_thresholds.py
python src/final\\\\\\\\\\\\\\\_strategy/evaluate\\\\\\\\\\\\\\\_strategy8\\\\\\\\\\\\\\\_test.py
```

**Reproduction and evaluation notes:**

* Confirm that the raw dataset files are in the expected locations before running the cleaning stage.
* Confirm that dependencies and script configuration match the environment used for the recorded results.
* Some scripts may overwrite existing artifacts. Back up any outputs you want to retain before rerunning experiments.
* Reruns may not reproduce the exact reported values if dependencies, random seeds, source data, split definitions, or configuration have changed.
* Keep the held-out test split out of feature selection, threshold tuning, model selection, and label-mapping decisions.

\---

## Interpretation and Limitations

1. **Offline evaluation only.** The repository evaluates prepared dataset records. It does not yet demonstrate live packet capture, streaming feature extraction, alert delivery, or production deployment.
2. **Target-label trade-off.** Merging Web Attack subclasses reduces label granularity. Strategy 8 does not identify the individual subtype within the merged class.
3. **Class imbalance.** Aggregate accuracy and weighted-F1 can obscure weaker performance on minority classes.
4. **Rare-class uncertainty.** The reported Heartbleed and Infiltration test supports are only 2 and 5 examples, respectively.
5. **Dataset representativeness.** Results on CICIDS2017 do not guarantee performance on modern traffic, other network environments, or unseen attack families.
6. **Operational costs differ.** False positives and false negatives have different consequences and should be evaluated for the intended use case.
7. **Reproducibility depends on configuration.** Dataset versions, label mapping, split definitions, dependency versions, feature ordering, and saved artifacts all affect the experiment.
8. **A classifier is not a complete security control.** Real-world use would require monitoring, alert triage, logging, incident response, access controls, and ongoing evaluation.
9. **Predictions are not certainty.** Offline test metrics are evidence about this evaluation setup, not a guarantee that a future flow will be classified correctly.

\---

## Future Work

Potential next steps include:

* Pin dependency versions and document a known-good environment for reproducibility.
* Populate exact Strategy 4–7 results from saved experiment reports.
* Generate confusion matrices and per-class metric plots from the saved reports or predictions.
* Evaluate on an independent dataset or a temporally separated traffic collection.
* Investigate calibration and operating thresholds using validation data only.
* Measure inference latency, memory consumption, and throughput before considering deployment.
* Add automated tests for feature order, label mapping, missing-value handling, and split integrity.
* Investigate dataset drift, unknown attack families, and confidence-based abstention.
* Measure false-positive workload and class-specific costs under a clearly defined operational scenario.
* Build an offline demonstration interface before attempting any live integration.

\---

## Dataset Attribution and License

**Dataset:** CICIDS2017  
**Publisher:** Canadian Institute for Cybersecurity, University of New Brunswick  
**Official source:** https://www.unb.ca/cic/datasets/ids-2017.html

Review the dataset's own terms and attribution requirements before redistribution. Raw and processed dataset files are intentionally excluded from this repository.

This repository contains an MIT `LICENSE` file. That license applies within its stated scope and does not override separate terms governing CICIDS2017 or third-party components.

\---

**Measure carefully. Preserve the evidence. Explain the trade-offs.**



