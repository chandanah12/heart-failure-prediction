# Heart Failure Prediction

Predicting heart disease from patient clinical data using Python, with a focus on data quality checks and honest model evaluation.

## Overview
- **Data:** Kaggle "Heart Failure Prediction" dataset, 918 patients and 12 features (age, sex, chest pain type, resting BP, cholesterol, max heart rate, ST slope, etc.)
- **Target:** `HeartDisease` (1 = disease, 0 = no disease). The dataset has 508 disease cases and 410 non-disease cases.
- **Tools:** Python, pandas, NumPy, matplotlib, seaborn, scikit-learn
- **Dataset link:** [https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction]. Download `heart.csv` and place it next to the notebook to run it.

## Key finding: hidden missing data
An initial comparison showed *lower* average cholesterol in heart disease patients, which is the opposite of what is medically expected. Investigating this showed:
- 172 rows had `Cholesterol = 0`, which is not physiologically possible and is a placeholder for missing data.
- 152 of those 172 rows were in the heart disease group, which pulled that group's average down and reversed the comparison.
- I replaced the zeros with `NaN` and filled them with the median of the valid values. After the fix, the expected pattern appeared (higher cholesterol in the disease group).
- I also checked `RestingBP` for the same problem and found no invalid zeros.

## Method
1. Exploratory analysis: missing values, distributions, correlations with the target, boxplots for outliers
2. Categorical analysis: crosstabs and count plots, for example chest pain type against heart disease
3. Preprocessing: one-hot encoding of categorical columns
4. Modeling: Logistic Regression and Random Forest on an 80/20 train/test split
5. Evaluation: accuracy, confusion matrix, recall and precision, then 5-fold cross-validation

## Results

| Model | Test accuracy (single split) | 5-fold CV accuracy | Missed disease cases (test set) |
|---|---|---|---|
| Logistic Regression | 86.4% | 84.2% (±4.7%) | 16 |
| Random Forest | 85.9% | 83.3% (±5.4%) | 14 |

- The test set had 184 patients, of whom 107 had heart disease. Always predicting "disease" would score about 58%, so both models learn real signal.
- Logistic Regression confusion matrix: 68 true negatives, 9 false positives, 16 false negatives, 91 true positives.
- The single-split result was slightly optimistic compared with cross-validation, so **about 84% is the more reliable estimate**.
- The two models are statistically indistinguishable here. The gap between them is under 1%, much smaller than the variation between splits.
- In a medical setting, **false negatives** (missed patients) are the costly error, so recall matters more than accuracy alone.

## Most important features
- Random Forest: `ST_Slope_Up`, `Oldpeak`, `MaxHR`
- Exploratory analysis: `ChestPainType_ASY` (asymptomatic chest pain) was strongly associated with heart disease, and `Oldpeak` and `Age` correlated positively with it.
- These features appear consistently across correlation analysis, crosstabs, and model importance.

## Limitations
- Small dataset (918 rows), so differences of 1-2% are not meaningful.
- The median used for cholesterol was calculated on the full dataset before the train/test split, which is a minor form of data leakage. A stricter pipeline would compute it on the training set only.
- No hyperparameter tuning or feature scaling was performed.
- This is a learning project, not a clinical tool.

## Next steps
- Use a scikit-learn Pipeline to remove the leakage
- Tune models and try gradient boosting
- Apply the same workflow to a credit-risk dataset

## How to run
