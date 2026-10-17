# SECOM: Fault Detection in Semiconductor Manufacturing

Feasibility study on predicting production failures from sensor and process measurements, based on the [SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) (UCI Machine Learning Repository). The work was carried out as the practical project of my M.Sc. in Data Science.

## Overview

The dataset contains 1,567 production runs with 590 sensor features and a pass/fail label. Only 104 runs (6.6 %) are failures, so the task is a strongly imbalanced binary classification problem with many missing and uninformative features.

The notebook `SECOM_en.ipynb` covers:

- **Exploratory analysis:** class distribution, missing values and their structure, constant and duplicated features, feature to label association, class separability (PCA), failure rate over time
- **Preprocessing:** custom `scikit-learn` transformers (missing value filter, variability filter, correlation filter), median imputation and scaling, all fitted on the training data only
- **Evaluation setup:** chronological 80/20 split (the test set contains the later runs, which mirrors predicting future runs from past data)
- **Models:** logistic regression (reference), gradient boosting, each with class weights and with SMOTE, plus a decision threshold tuned on the MCC
- **Robustness:** bootstrap confidence intervals for MCC and F1, a feature set ablation, a cost comparison against "never check" and "always check", and a diagnostic on how the preprocessing thresholds depend on the data basis
- **Dashboard prototype:** an interactive overview with a single case view for runs from the test set

Because of the imbalance, accuracy is not informative here. The notebook focuses on F1 of the failure class, the Matthews correlation coefficient (MCC) and ROC AUC.

## Main findings

On the chronological test set (314 runs, 17 failures) the reference model (balanced logistic regression on 13 correlation-selected features) reaches F1 = 0.30, MCC = 0.26 (95 % bootstrap CI approx. 0.09 to 0.42) and ROC AUC = 0.71. Gradient boosting and the SMOTE variants do not improve on this in a reliable way, and the confidence intervals of the variants overlap considerably. With only 17 failures in the test set, the results are best read as a feasibility indication, not as a production-ready detector.

## Repository structure

```
.
├── SECOM_en.ipynb      # analysis and models
├── README.md
└── secom/              # not included, see below
    ├── secom.data
    ├── secom_labels.data
    └── secom.names
```

## Getting started

1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/179/secom) and place `secom.data`, `secom_labels.data` and `secom.names` in a folder `secom/` next to the notebook. Please check the UCI page for the licence and citation terms of the data.
2. Install the dependencies (Python 3.13 and scikit-learn 1.8 were used):

```bash
   pip install numpy pandas matplotlib seaborn missingno scipy scikit-learn imbalanced-learn ipywidgets jupyterlab
```

3. Open `SECOM_en.ipynb` and run all cells. A fixed random seed (`SEED = 33`) is used, but results can still differ slightly across library versions.

The notebook is stored without outputs. The last cell calls the dashboard for run 1327 so that it is visible without interaction; the dropdown lets you inspect other test runs.

## Reference

McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository.