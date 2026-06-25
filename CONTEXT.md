# CONTEXT.md - Machine Learning Project Part B

## Project Identity

This repository contains **Machine Learning 364-1-1811, Project Part B**, Group 3, using the Airline Satisfaction dataset.

The submitted code artifact is:

```text
Airline_Project_Part_B_Group3.ipynb
```

The notebook runs from the raw project files, performs corrected Part A cleaning internally, trains and compares the required models, selects a final model, and exports predictions.

## Current Scope Decision

The official `part b instructions.md` still mentions K-Means and Naive Bayes. The lecturer subsequently removed both from the required scope. This documented lecturer decision takes precedence for the current project.

F1 is the leading metric for hyperparameter tuning and final model selection. The methodological reason is that F1 balances Precision and Recall for the positive class (`satisfied`), so it evaluates both the ability to identify satisfied passengers and the reliability of positive predictions. Accuracy is still reported because the classes are moderately balanced, about 56% / 44%, but it is not the primary selection metric.

Required modeling scope:

- Decision Tree.
- MLP / Neural Network.
- Hyperparameter tuning and Cross-Validation.
- Model comparison and final selection using F1.
- Final test prediction export.
- Optional subgroup analysis for bonus consideration.

Do not deduct for the absence of K-Means or Naive Bayes, and do not reintroduce them unless explicitly requested.

## Required Files

Raw notebook inputs in the project folder:

```text
Xy_train.csv
X_test.csv
```

Supporting material:

```text
part b instructions.md
ml_course_context_decisions_dt_nn.md
AirlineSatisfaction Dataset.docx
```

Generated artifacts:

```text
airline_G3_ytest.xlsx
Bonus_Table.png
```

The raw CSV files must never be modified. No intermediate cleaned CSV is required.

## Target And Eligibility Filtering

The target column is detected automatically by comparing train and test schemas. In the current data it is `satisfaction`.

Binary mapping:

```python
TARGET_MAP = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}
```

Only deterministic eligibility operations occur before the train-validation split:

- Remove the one row with missing target.
- Remove the one training row with an invalid non-missing free-text `Class`.
- Keep missing or `Unknown` Class values for treatment inside the cleaner.
- Do not compute a median, mode, grouped statistic, category vocabulary, or scaling parameter before the split.

Verified row counts:

```text
Raw labeled rows: 9000
Eligible labeled rows: 8998
Training rows: 7198
Validation rows: 1800
Final test rows: 1000
```

The stratified split uses `test_size=0.2` and `random_state=42`. Class proportions are 56.36% / 43.64% in training and 56.33% / 43.67% in validation.

## Leakage-Safe Cleaning

Cleaning is implemented by `AirlinePartACleaner`, an sklearn-compatible transformer derived from `BaseEstimator` and `TransformerMixin`.

`fit()` learns only from the rows supplied to it:

- Median `Gate location` after invalid values are converted to missing.
- `Leg room service` medians by Class + Type of Travel, then by Class, then globally.
- Median `Age` after values outside 0-110 are converted to missing.
- `Flight Distance` medians by Class + Type of Travel, then by Class, then globally, after negative values are converted to missing.
- Remaining numeric medians.
- Remaining categorical modes.

`transform()` applies stored values without recomputing them and never removes rows.

Additional behavior:

- Missing and `Unknown` Class values become `Business`.
- Invalid final-test Class values, if present, are treated as missing and become `Business`.
- `Plane colors` is dropped.
- `Age_Category`, `Flight_Type`, and `Total_Service_Score` are created.
- `Flight_Type` uses fixed bins `[0, 1000, 3000, np.inf]`.
- The transformed feature set contains 23 columns and no missing values.

Verified holdout-cleaner statistics learned from the 7,198 training rows only:

```text
Gate Location median: 3.0
Age median: 41.0
Global Leg Room median: 3.0
Global Flight Distance median: 853.0
```

Verified full-training cleaner statistics learned from all 8,998 eligible labeled rows:

```text
Gate Location median: 3.0
Age median: 41.0
Global Leg Room median: 3.0
Global Flight Distance median: 853.5
```

No `fit` call is made on `X_valid` or `X_test`.

## Decision Tree Results

Modeling process:

```text
AirlinePartACleaner
-> dynamic numeric/categorical selectors
-> numeric passthrough + OneHotEncoder(handle_unknown="ignore")
-> DecisionTreeClassifier
```

Decision Tree tuning uses:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

The one-dimensional plots report Mean CV F1, not validation-set performance.

Verified one-dimensional F1 results:

```text
Best max_depth: 8; Mean CV F1: 0.8585
Best ccp_alpha: 0.000857; Mean CV F1: 0.8708
Best min_samples_split: 97; Mean CV F1: 0.8637
Best min_samples_leaf: 15; Mean CV F1: 0.8688
```

The combined search grid remains:

```text
max_depth: [6, 8, 9, 10, 12]
min_samples_split: [30, 60, 80, 100]
min_samples_leaf: [15, 25, 35, 45]
ccp_alpha: 10 percentile-sampled values from the X_train pruning path
```

Verified selected Decision Tree configuration:

```text
max_depth: 12
ccp_alpha: 0.00038492671497453553
min_samples_split: 30
min_samples_leaf: 15
Best 5-fold Mean CV F1: 0.8711
```

Full tree baseline:

```text
Train Accuracy/F1: 1.0000 / 1.0000
Validation Accuracy: 0.8478
Validation Precision: 0.8216
Validation Recall: 0.8321
Validation F1: 0.8268
```

Selected Decision Tree metrics:

```text
Train Accuracy: 0.9132
Train Precision: 0.9141
Train Recall: 0.8841
Train F1: 0.8989

Validation Accuracy: 0.8828
Validation Precision: 0.8808
Validation Recall: 0.8461
Validation F1: 0.8631
Confusion matrix: [[924, 90], [121, 665]]
```

The manual path example remains validation position 10, original index 4267, and is correctly classified as Satisfied. It reaches leaf 62 in the rebuilt tree.

Top five Decision Tree feature importances:

```text
Type of Travel_Personal Travel: 0.2713
Inflight wifi service: 0.1872
Total_Service_Score: 0.1841
Customer Type_Loyal Customer: 0.1338
Cleanliness: 0.0795
```

## MLP Results

Modeling process:

```text
AirlinePartACleaner
-> dynamic numeric/categorical selectors
-> StandardScaler for numeric + OneHotEncoder for categorical
-> MLPClassifier
```

The encoded MLP input has 33 features.

MLP tuning uses:

```python
StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
```

Three folds are documented as a compromise between estimate stability and neural-network runtime. The search keeps the existing architecture space and two alpha values, but now optimizes F1.

Default MLP baseline:

```text
Train Accuracy: 0.9931
Train Precision: 0.9924
Train Recall: 0.9917
Train F1: 0.9920

3-fold CV Accuracy: 0.8808
3-fold CV Precision: 0.8714
3-fold CV Recall: 0.8529
3-fold CV F1: 0.8619
```

The default MLP reaches `max_iter=500` without full convergence. It is retained only as a baseline.

Verified selected MLP configuration:

```text
hidden_layer_sizes: (150,)
activation: relu
alpha: 0.0001
learning_rate_init: 0.001
Best 3-fold Mean CV F1: 0.8739
```

Selected MLP metrics:

```text
Train Accuracy: 0.9225
Train Precision: 0.9151
Train Recall: 0.9064
Train F1: 0.9107

Validation Accuracy: 0.8856
Validation Precision: 0.8826
Validation Recall: 0.8511
Validation F1: 0.8666
Confusion matrix: [[925, 89], [117, 669]]
```

## Model Comparison And Selection

Final selection prioritizes validation F1 because it balances precision and recall for the positive class.

Comparison:

```text
Decision Tree: Accuracy 0.8828, Precision 0.8808, Recall 0.8461, F1 0.8631, gap 0.0304
MLP:           Accuracy 0.8856, Precision 0.8826, Recall 0.8511, F1 0.8666, gap 0.0369
```

The selected final model is the tuned MLP. The MLP has higher validation F1 and slightly higher validation Accuracy, Precision, and Recall. The Decision Tree remains more interpretable and has a slightly smaller generalization gap.

## Bonus Subgroup Analysis

The subgroup analysis uses the selected MLP's validation predictions. Grouping columns are taken from `X_valid` after applying the selected model's already-fitted cleaner, so missing category values are handled consistently.

The analysis is descriptive only and is not used for further tuning.

Verified subgroup findings:

```text
Personal Travel: 546 records, 56 satisfied, Accuracy 0.9158, Satisfied Recall 0.4107, 33 false negatives
Business Travel: 1254 records, 730 satisfied, Accuracy 0.8724, Satisfied Recall 0.8849, 84 false negatives
Disloyal Customer: 320 records, 72 satisfied, Accuracy 0.8344, Satisfied Recall 0.5278, 34 false negatives
Loyal Customer: 1480 records, 714 satisfied, Accuracy 0.8966, Satisfied Recall 0.8838, 83 false negatives
```

`Bonus_Table.png` is regenerated from the current subgroup plot when the notebook runs.

## Final Prediction Export

The final MLP training and prediction process is rebuilt from the selected hyperparameters and fitted to all 8,998 eligible raw labeled rows. Its cleaner learns only from those rows. The untouched 1,000-row final test frame is passed to `predict` in original order.

Verified output file:

```text
airline_G3_ytest.xlsx
Rows: 1000
Columns: 1
Column name: target
Missing values: 0
Class 0 predictions: 595 (59.5%)
Class 1 predictions: 405 (40.5%)
```

The file is read back and compared with the in-memory submission. If Excel locks the target file, the notebook reports an environmental issue and validates a temporary Excel file instead.

Compared with the snapshot before switching tuning to F1:

```text
Changed predictions: 51
0 -> 1 changes: 18
1 -> 0 changes: 33
```

## Verified Notebook Status

Latest full execution:

- 39 code cells executed with sequential counts 1 through 39.
- No cell errors.
- All transformed train, validation, and test frames contain 23 features and zero missing values.
- No fitting on validation or final test data.
- No use of `X_valid` inside Decision Tree or MLP hyperparameter tuning.
- Decision Tree and MLP tuning both optimize F1.
- Final model selection uses Validation F1.
- Final test row count and order preserved.
- Final Excel export passed all structural checks.
- Raw source CSV hashes remain unchanged.

Environment-only messages observed during execution:

- The default MLP emitted a convergence warning at 500 iterations.
- Joblib on Windows emitted resource-tracker cleanup messages after parallel work completed.
- These messages did not produce notebook cell errors or alter the successful export.

## Presentation And Writing Rules

- Markdown explanations and conclusions remain in Hebrew with RTL wrappers.
- Code and code-generated table/plot labels remain in English.
- Conclusions must match saved outputs exactly.
- Explain that F1 is the leading metric because it balances Precision and Recall for the positive class.
- Explain that Accuracy remains informative because the class distribution is moderately balanced.
- Keep the notebook suitable for classroom presentation and final report use.
