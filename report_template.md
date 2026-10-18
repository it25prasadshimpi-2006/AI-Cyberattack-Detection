# Design and Implementation of an AI-Based Cyberattack Detection System Using Machine Learning

## Title page
- College / Department:
- Student name and roll number:
- Guide:
- Academic year:

## Abstract
Summarize the problem, dataset, models, measured results, and dashboard. Fill in actual results after running the experiment.

## Chapter 1: Introduction
### 1.1 Background
### 1.2 Problem statement
### 1.3 Objectives
### 1.4 Scope and limitations

## Chapter 2: Literature Review
Summarize relevant intrusion detection and ML research. Cite each source correctly.

## Chapter 3: Dataset
- Dataset: UNSW-NB15
- Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Dataset version and files used:
- Feature descriptions:
- Class distribution (include a chart):
- Label definition: 0 = Normal, 1 = Attack (verify for your downloaded version)

## Chapter 4: Methodology
- Data loading and quality checks
- Missing-value and infinity handling
- Categorical encoding
- Numeric scaling
- Train/test split strategy
- Algorithms: Logistic Regression and Random Forest
- Metrics: accuracy, precision, recall, F1-score, ROC-AUC

## Chapter 5: System Design and Implementation
- Architecture diagram
- Software and hardware requirements
- Folder structure
- Preprocessing pipeline
- Training module
- Streamlit dashboard
- Screenshots

## Chapter 6: Results and Discussion
Insert actual output from `results/model_comparison.csv`.
Discuss:
- Which model had the highest F1-score?
- What was the attack recall?
- How many false positives and false negatives occurred?
- What trade-offs exist?
- What are the limits of generalizing to live networks?

## Chapter 7: Conclusion and Future Scope
Discuss findings supported by your measurements. Future scope can include multiclass prediction, SHAP, independent-dataset testing, and authorized lab traffic analysis.

## References
1. UNSW-NB15 dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/

## Appendix
- Commands used
- Additional plots
- Selected code snippets
