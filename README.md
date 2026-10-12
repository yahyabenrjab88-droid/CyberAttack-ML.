\# CICIDS2017 Network Intrusion Detection Using Machine Learning



A machine-learning project for multi-class network intrusion detection using the CICIDS2017 dataset. This repository documents the complete experimental journey: raw-data investigation, cleaning, preprocessing, dataset splitting, baseline modeling, successive experiments, error analysis, and final held-out test evaluation.



The project prioritizes \*\*data quality, leakage prevention, reproducibility, class-level evaluation, and honest experimental reporting\*\* rather than relying on accuracy alone.



\## Project Overview



The CICIDS2017 dataset contains benign network traffic and multiple categories of malicious traffic. Although it is widely used for intrusion-detection research, its class imbalance, duplicate records, suspicious feature values, and overlapping attack characteristics require careful investigation.



This project began with auditing and cleaning the dataset before building a consistent preprocessing pipeline. A baseline model was then used to identify difficult attack classes, especially Web Attack Brute Force and Web Attack XSS. Multiple approaches were investigated before selecting a 13-class HistGradientBoosting model that merges the three Web Attack labels into one class.



\### Final held-out test results



| Metric | Result |

|---|---:|

| Macro-F1 | \*\*96.61%\*\* |

| Accuracy | \*\*99.54%\*\* |

| Weighted-F1 | \*\*99.55%\*\* |



These figures correspond to the saved held-out test evaluation. Macro-F1 is emphasized because it gives each class equal weight, making it more informative than accuracy alone when classes are imbalanced.



The results should be interpreted alongside the per-class metrics, class supports, and limitations documented below.



\## Objectives



\- Investigate data quality before model training.

\- Detect and document duplicates, conflicting labels, invalid values, and suspicious feature patterns.

\- Build a repeatable cleaning, splitting, and preprocessing workflow.

\- Prevent information leakage during imputation and model development.

\- Establish a baseline and preserve subsequent experiments, including unsuccessful approaches.

\- Evaluate models using Macro-F1, accuracy, weighted-F1, confusion matrices, and per-class precision and recall.

\- Preserve trained models, predictions, reports, and experiment configurations for future analysis.

\- Establish a foundation for a future local network intrusion detection platform.



\## Dataset



\*\*Dataset:\*\* \[CICIDS2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)



The original dataset contains labeled network-flow records collected across multiple days and attack scenarios. The project investigates benign traffic and attacks such as DoS, DDoS, Bot, PortScan, brute-force attacks, and web attacks.



The original Web Attack labels include:



\- Web Attack — Brute Force

\- Web Attack — XSS

\- Web Attack — SQL Injection



For the final modeling strategy, these three labels are merged into a single \*\*Web Attack\*\* class.



\### Dataset availability



Raw and processed datasets are intentionally excluded from this repository because of their size and dataset redistribution considerations. The repository retains the source code and generated reports needed to document the work.



Obtain the dataset from its official source and place the files in the expected local data directory before reproducing the pipeline.



\## Methodology



\### 1. Raw-data investigation



The first phase examined the original CSV files before committing to a modeling approach.



Investigations covered:



\- Raw file and column consistency.

\- Label distributions across files.

\- Feature validity and suspicious numerical values.

\- Missing values and positive or negative infinity.

\- Negative flow statistics, header lengths, and duration-related values.

\- Inter-arrival-time anomalies.

\- Subflow inconsistencies.

\- Exact and cross-file duplicate feature vectors.

\- Duplicate records associated with conflicting labels.



The investigation reports preserve examples and summaries instead of silently discarding every suspicious observation. This makes the cleaning decisions easier to review.



Relevant outputs are stored under `reports/data\_quality/`, `reports/duplicates/`, and `reports/investigations/`.



\### 2. Cleaning



The cleaning stage consolidates the raw data into a cleaned dataset and records quality-related findings.



The objective is to establish a consistent dataset while retaining evidence about the original data problems.



Important distinctions include:



\- Missing values are not automatically replaced with zero.

\- Positive and negative infinity are converted to missing values for subsequent handling.

\- Duplicate and conflicting-label observations are investigated rather than treated as interchangeable without verification.

\- Cleaning decisions are documented in reports.



The principal cleaned artifact is `data/processed/cleaned.parquet`.



\### 3. Train, validation, and test splitting



The cleaned dataset is divided into three separate subsets:



\- \*\*Training set:\*\* used to fit models and training-dependent preprocessing.

\- \*\*Validation set:\*\* used to compare experiments and tune modeling decisions.

\- \*\*Test set:\*\* reserved for final evaluation.



The project uses a 70% / 15% / 15% train-validation-test split and retains split-distribution audit reports.



The held-out test set must not be used to select model features, tune hyperparameters, or choose decision thresholds. Final test metrics are meaningful only when the test set remains separate from those decisions.



\### 4. Preprocessing and leakage prevention



The preprocessing pipeline prepares the data for consistent model inputs.



Key decisions:



1\. Positive and negative infinity values are converted to missing values.

2\. Median imputation is fitted using the training data only.

3\. The fitted imputation values are reused for validation and other applicable evaluation data.

4\. Missing-value indicators are retained for selected features.

5\. Model feature columns are kept consistent across training and inference.



The current model input consists of \*\*68 numeric features\*\*. Each processed dataset may also contain the `source\_file` metadata column and the `Label` target column. Neither should be included as an ordinary model feature.



Missing indicators are retained for:



\- `Flow Bytes/s`

\- `Flow IAT Min`

\- `Fwd IAT Min`



A separate `Flow Packets/s\_missing` indicator is omitted because it duplicates the `Flow Bytes/s\_missing` indicator.



These decisions are intended to prevent leakage, avoid artificial zero values, and preserve a stable feature schema.



\## Experimental Journey



The experiments were driven by observed model errors rather than accuracy alone. A recurring challenge was the overlap between Web Attack Brute Force and Web Attack XSS.



The earlier BF/XSS experiments retain their original results and model artifacts wherever available.



\### Strategies 1–3: Baseline and initial specialist approaches



| Experiment | Validation Macro-F1 | BF/XSS mutual confusion | Outcome |

|---|---:|---:|---|

| Strategy 1 — baseline HistGradientBoosting | 85.8601% | 108 | Baseline reference |

| Strategy 2 — unconditional BF/XSS specialist | 86.3277% | 106 | Rejected |

| Strategy 3 — calibrated, gated specialist, threshold 0.55 | 84.77% | 88 | Rejected |



The baseline achieved high overall accuracy but exposed weaknesses in minority and overlapping attack classes. Its reported BF F1 was 0.7383, while XSS F1 was 0.4464.



Strategy 2 improved Macro-F1 slightly, but mutual BF/XSS confusion decreased only marginally. It was not a sufficient solution.



Strategy 3 reduced mutual confusion further, but its XSS F1 deteriorated. This illustrates why reducing one type of error does not necessarily improve the overall classifier.



\### Strategy 4 — Hard-example-weighted specialist cascade



This experiment introduced hard-example weighting into a specialist approach intended to distinguish BF from XSS more effectively.



The goal was to emphasize difficult training examples rather than treating all errors identically. The saved primary and specialist models, comparison reports, and training summary preserve the experiment.



The approach is retained as part of the research history rather than presented as a universally successful solution.



\### Strategy 5 — Out-of-fold stacked meta-learner



This experiment used out-of-fold (OOF) predictions to train a meta-learner for the BF/XSS problem.



The objective was to learn when and how the specialist predictions should influence the primary classifier, using out-of-fold predictions to reduce leakage from in-sample predictions.



The repository retains the primary model, logistic-regression meta-model, feature scaler, class ordering, and experiment reports.



\### Strategy 6 — SMOTE-Tomek specialist



This experiment introduced SMOTE-Tomek resampling to address the imbalance in the BF/XSS specialist's training data.



SMOTE generates synthetic minority-class examples, while Tomek-link removal can remove selected overlapping observations. Resampling was applied as an experimental approach, not assumed to guarantee better generalization.



Before-and-after reports are retained to document the effect on classification performance.



\### Strategy 7 — OOF stacking with SMOTE-Tomek



Strategy 7 combined out-of-fold meta-learning with SMOTE-Tomek resampling.



It explored whether combining a learned decision layer with resampling could improve difficult BF/XSS classifications. The repository preserves the model, meta-model, scaler, class ordering, and evaluation reports.



\### Strategy 8 — Merged Web Attack classification



The final selected modeling approach changes the target formulation rather than continuing to force the classifier to separate highly overlapping Web Attack subcategories.



The labels Web Attack Brute Force, Web Attack XSS, and Web Attack SQL Injection are merged into one Web Attack class. The model is a HistGradientBoosting classifier trained on the consistent 68-feature input.



This produces a \*\*13-class classification task\*\*.



Saved artifacts include:



\- `models/strategy8\_web\_attack\_merged/hgb\_13class\_model.joblib`

\- `models/strategy8\_web\_attack\_merged/feature\_columns.joblib`

\- `models/strategy8\_web\_attack\_merged/class\_order.joblib`



The feature-column and class-order files are important for ensuring that later inference uses the same feature ordering and class mapping as training.



\### Strategy 8 validation results



The saved validation results report approximately:



| Metric | Validation result |

|---|---:|

| Macro-F1 | 96.5579% |

| Accuracy | 99.16% |

| Weighted-F1 | 99.16% |



These results should not be confused with the final held-out test evaluation.



\### Strategy 8b — BENIGN versus DoS Hulk threshold investigation



After selecting the merged Web Attack model, an additional experiment investigated the decision threshold affecting BENIGN and DoS Hulk predictions.



The experiment is a threshold-analysis step, not a replacement for the complete 13-class model. Its comparison results are retained under `results/strategy8b\_hulk\_benign\_thresholds/`.



The validation threshold experiment reported Macro-F1 of approximately 96.6581%. This validation result is not, by itself, proof of improved test generalization.



\### Final held-out test evaluation



The saved evaluation under `results/final\_test\_evaluation\_threshold025/` records:



| Metric | Test result |

|---|---:|

| Macro-F1 | 96.6124% |

| Accuracy | 99.54% |

| Weighted-F1 | 99.55% |



The test report also contains per-class precision, recall, F1, support, a confusion matrix, and test predictions.



Reported class-level observations include:



\- \*\*Bot:\*\* precision 74.62%, recall 98.64%, F1 84.96%.

\- \*\*Web Attack:\*\* precision 93.33%, recall 98.47%, F1 95.83%.

\- \*\*Heartbleed:\*\* support of 2 test observations.

\- \*\*Infiltration:\*\* support of 5 test observations.



The perfect scores reported for Heartbleed and Infiltration should not be interpreted as robust evidence of generalization because the test support is extremely small.



\## Results and Interpretation



The project demonstrates why intrusion-detection models should be evaluated beyond accuracy.



\- Macro-F1 helps expose performance differences across imbalanced classes.

\- Confusion matrices reveal which attack categories are mistaken for one another.

\- Per-class precision and recall reveal trade-offs that a single aggregate metric hides.

\- A change that improves one class or reduces one type of error may worsen another.

\- Merging overlapping labels can improve the usefulness of a classifier when the original label distinction is difficult to learn reliably.



The final model is the current selected approach for this dataset and label formulation. The experiments are not evidence that the same performance will be achieved on different networks, capture environments, or unseen attack families.



\## Repository Structure



```text

CyberAttack-ML/

├── data/

│   ├── raw/                         # Local source CSV files; not tracked

│   └── processed/                   # Local cleaned/split data; not tracked

├── models/

│   ├── historical/                  # Models from previous experiments

│   └── strategy8\_web\_attack\_merged/ # Selected 13-class model and schema

├── reports/

│   ├── data\_quality/                # Data and label audits

│   ├── duplicates/                  # Duplicate and conflict investigations

│   ├── investigations/              # Feature anomalies and diagnostics

│   └── preprocessing\_split/         # Preprocessing and split audits

├── results/

│   ├── baseline/

│   ├── bfxss\_gated\_calibrated\_cascade/

│   ├── bfxss\_specialist\_cascade/

│   ├── final\_test\_evaluation\_threshold025/

│   ├── strategy4\_hardweighted\_specialist\_cascade/

│   ├── strategy5\_oof\_stacked\_metalearner/

│   ├── strategy6\_smote\_tomek/

│   ├── strategy7\_oof\_smote\_tomek/

│   ├── strategy8\_web\_attack\_merged/

│   └── strategy8b\_hulk\_benign\_thresholds/

├── src/

│   ├── data\_pipeline/               # Cleaning, splitting, preprocessing

│   ├── final\_strategy/              # Strategy 8 and final evaluation tools

│   ├── historical\_experiments/      # Earlier training experiments

│   ├── investigations/              # Data-quality investigations

│   └── validation/                  # Pipeline validation scripts

├── requirements.txt

├── requirements\_before\_strategy6.txt

├── src\_inventory.txt

├── .gitignore

├── LICENSE

└── README.md

```



\## Installation and Reproduction



The commands below describe the intended workflow. They have not been executed as part of preparing this README. The repository's scripts may be computationally expensive, and rerunning a script can overwrite existing artifacts.



\### 1. Set up the environment



From PowerShell:



```powershell

Set-Location C:\\CyberAttack-ML



py -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



python -m pip install --upgrade pip

pip install -r requirements.txt

```



Download the CICIDS2017 dataset from its official source and place the raw CSV files in `data/raw/`.



\### 2. Run the data pipeline



```powershell

python src/data\_pipeline/clean.py

python src/data\_pipeline/split.py

python src/data\_pipeline/preprocess.py

```



These stages prepare the cleaned dataset, create the train/validation/test splits, and generate the preprocessed training and validation artifacts used by the modeling workflow.



\### 3. Validate the pipeline outputs



```powershell

python src/validation/audit\_cleaned.py

python src/validation/audit\_splits.py

python src/validation/audit\_preprocessed.py

```



Review the reports under `reports/` before proceeding. Confirm the expected feature columns, label distributions, and preprocessing behavior.



\### 4. Reproduce the experiments



Run experiments individually, in the intended order, after confirming that their required input files are available.



```powershell

\# Baseline

python src/historical\_experiments/train\_baseline.py



\# Strategies 2–3: BF/XSS specialist approaches

python src/historical\_experiments/train\_hgb\_bfxss\_specialist\_cascade.py

python src/historical\_experiments/train\_hgb\_bfxss\_gated\_calibrated\_cascade.py



\# Additional hard-example and hard-weighting investigations

python src/historical\_experiments/train\_hgb\_bfxss\_hard\_example\_specialist.py

python src/historical\_experiments/train\_hgb\_bfxss\_hardweighted\_specialist\_cascade.py



\# Strategies 5–7: OOF stacking and resampling

python src/historical\_experiments/train\_hgb\_bfxss\_oof\_stacked\_metalearner.py

python src/historical\_experiments/train\_hgb\_bfxss\_smote\_tomek\_specialist.py

python src/historical\_experiments/train\_hgb\_bfxss\_oof\_smote\_tomek.py

```



Compare the resulting reports and model artifacts with the saved historical results. The original experiment scripts and saved reports should be consulted for each experiment's exact configuration.



\### 5. Train the selected Strategy 8 model



```powershell

python src/final\_strategy/train\_hgb\_web\_attack\_merged.py

```



This script trains the merged Web Attack classifier and saves the model, feature-column list, class ordering, and validation reports.



\### 6. Investigate thresholds and errors



```powershell

python src/final\_strategy/analyze\_strategy8\_errors.py

python src/final\_strategy/strategy8b\_hulk\_benign\_thresholds.py

```



Use validation results for threshold exploration and error analysis. Do not select a threshold based on held-out test performance.



\### 7. Final test evaluation



```powershell

python src/final\_strategy/evaluate\_strategy8\_test.py

```



This command uses the held-out test data. It is included to document the evaluation workflow, not as a step that must be repeated during routine development. Preserve the test set for final evaluation rather than using it to tune the model.



\*\*Reproducibility note:\*\* Exact dependency versions, random seeds, split settings, and experiment-specific parameters should be taken from the scripts and saved reports. The commands above describe the workflow but do not guarantee that rerunning every historical experiment will reproduce identical outputs across software versions or environments.



\## Limitations



1\. \*\*Class imbalance:\*\* Some attack categories have far fewer examples than others.

2\. \*\*Rare-class uncertainty:\*\* Perfect metrics on a handful of Heartbleed or Infiltration examples are not sufficient to establish reliable detection.

3\. \*\*Overlapping attacks:\*\* Similar flow characteristics can make certain attack categories difficult to distinguish.

4\. \*\*Label merging:\*\* Merging the Web Attack subclasses simplifies the target task but sacrifices the ability to distinguish those three subtypes.

5\. \*\*Dataset generalization:\*\* Results on CICIDS2017 do not establish performance on modern production networks or unseen attacks.

6\. \*\*False positives and false negatives:\*\* The model may misclassify benign traffic or fail to identify malicious traffic.

7\. \*\*Offline evaluation:\*\* This repository currently documents dataset-based classification. It is not yet a complete real-time network monitoring or packet-capture platform.



\## Future Work



\- Build a local packet-capture and flow-feature extraction pipeline.

\- Verify that live feature extraction exactly matches the 68 training features.

\- Test the model on authorized lab traffic without uploading packet data to external services.

\- Add a local dashboard for predictions, alert history, and class-level statistics.

\- Evaluate generalization on independent datasets and different traffic environments.

\- Improve evaluation for rare classes and investigate false-positive rates under realistic traffic conditions.



\## Responsible Use



Use the model for research, learning, and authorized defensive monitoring. Predictions are probabilistic classifications, not definitive proof that an attack has occurred. Validate detections before taking disruptive actions.



\## License and Dataset Attribution



The CICIDS2017 dataset is maintained by the Canadian Institute for Cybersecurity. Consult the official dataset source for its terms of use and attribution requirements.



The repository's `LICENSE` file must contain the chosen project license before a public release can make a clear licensing claim. Do not assume that the project's license grants rights to redistribute the original dataset.



\## Acknowledgments



Thanks to the Canadian Institute for Cybersecurity for providing CICIDS2017 and to the broader intrusion-detection research community for the tools and methods that support reproducible ML experimentation.
