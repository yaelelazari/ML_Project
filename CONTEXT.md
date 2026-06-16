# CONTEXT.md - Machine Learning Project Part B

## Project Identity

This repository contains **Machine Learning 364-1-1811, Project Part B**, Group 3, using the Airline Satisfaction dataset.

The submitted code artifact is:

```text
Airline_Project_Part_B_Group3.ipynb
```

The notebook must run from top to bottom from the raw project files, perform the corrected Part A cleaning internally, train and compare the required models, select a final model, and export predictions.

## Current Scope Decision

The official `part b instructions.md` still mentions K-Means and Naive Bayes. The lecturer subsequently removed both from the required scope. This documented lecturer decision takes precedence for the current project.

Required modeling scope:

- Decision Tree.
- MLP / Neural Network.
- Hyperparameter tuning and Cross-Validation.
- Model comparison and final selection.
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

Cleaning is implemented by the sklearn-compatible transformer `AirlinePartACleaner`, derived from `BaseEstimator` and `TransformerMixin`.

`fit()` learns only from the rows supplied to it:

- Median `Gate location` after invalid values are converted to missing.
- `Leg room service` medians by Class + Type of Travel, then by Class, then globally.
- Median `Age` after values outside 0-110 are converted to missing.
- `Flight Distance` medians by Class + Type of Travel, then by Class, then globally, after negative values are converted to missing.
- Remaining numeric medians.
- Remaining categorical modes.

`transform()` applies the stored values without recomputing them and never removes rows.

Additional cleaning behavior:

- Missing and `Unknown` Class values become `Business` inside the transformer.
- Invalid test Class values, if present, are treated as missing and become `Business`.
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

## Decision Tree Pipeline And Results

Pipeline:

```text
AirlinePartACleaner
-> dynamic numeric/categorical selectors
-> numeric passthrough + OneHotEncoder(handle_unknown="ignore")
-> DecisionTreeClassifier
```

Decision Tree Cross-Validation uses:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

The one-dimensional plots report Mean CV Accuracy, not validation-set accuracy.

Verified one-dimensional results:

```text
Best max_depth: 8; Mean CV Accuracy: 0.8789
Best ccp_alpha: 0.000857; Mean CV Accuracy: 0.8889
Best min_samples_split: 97; Mean CV Accuracy: 0.8832
Best min_samples_leaf: 15; Mean CV Accuracy: 0.8871
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
Best 5-fold Mean CV Accuracy: 0.8891
```

Full tree baseline:

```text
Train fit Accuracy: 1.0000
5-fold CV Accuracy: 0.8569
5-fold CV Precision: 0.8393
5-fold CV Recall: 0.8313
5-fold CV F1: 0.8352
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
Generalization gap: 0.0304
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

## MLP Pipeline And Results

Pipeline:

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

Three folds are documented as a compromise between estimate stability and neural-network runtime. The search keeps the existing architecture space and two alpha values.

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
hidden_layer_sizes: (50, 25, 10)
activation: relu
alpha: 0.001
learning_rate_init: 0.001
Best 3-fold Mean CV Accuracy: 0.8927
```

Selected MLP metrics:

```text
Train Accuracy: 0.9398
Train Precision: 0.9436
Train Recall: 0.9169
Train F1: 0.9301

Validation Accuracy: 0.8828
Validation Precision: 0.8778
Validation Recall: 0.8499
Validation F1: 0.8636
Generalization gap: 0.0571
Confusion matrix: [[921, 93], [118, 668]]
```

Accuracy is the primary metric because the class balance is moderate, about 56% / 44%. Precision, Recall, and F1 are also reported to detect class-specific weaknesses.

## Model Comparison And Selection

The Decision Tree and MLP have exactly the same validation Accuracy: `0.882777...`, displayed as `0.8828`.

Comparison:

```text
Decision Tree: Precision 0.8808, Recall 0.8461, F1 0.8631, gap 0.0304
MLP:           Precision 0.8778, Recall 0.8499, F1 0.8636, gap 0.0571
```

Because validation Accuracy is tied, validation F1 is used as the tie-breaker. The selected final model is the tuned MLP. The notebook explicitly notes that the margin is very small and that the Decision Tree remains more interpretable and has a smaller generalization gap.

## Bonus Subgroup Analysis

The subgroup analysis uses the selected MLP's validation predictions. Grouping columns are taken from `X_valid` after applying the selected pipeline's already-fitted cleaner, so missing category values are handled consistently and no extra `NaN` subgroup remains.

The analysis is descriptive only and is not used for further tuning.

Verified subgroup findings after applying the selected cleaner:

```text
Personal Travel: 546 records, 56 satisfied, Accuracy 0.9121, Satisfied Recall 0.4643, 30 false negatives
Business Travel: 1254 records, 730 satisfied, Accuracy 0.8700, Satisfied Recall 0.8795, 88 false negatives
Disloyal Customer: 320 records, 72 satisfied, Accuracy 0.8563, Satisfied Recall 0.6111, 28 false negatives
Loyal Customer: 1480 records, 714 satisfied, Accuracy 0.8885, Satisfied Recall 0.8739, 90 false negatives
```

`Bonus_Table.png` is regenerated from the current subgroup plot when the notebook runs.

## Final Prediction Export

The final MLP pipeline is rebuilt from the selected hyperparameters and fitted to all 8,998 eligible raw labeled rows. Its cleaner learns only from those rows. The untouched 1,000-row final test frame is passed to `predict` in original order.

Verified output file:

```text
airline_G3_ytest.xlsx
Rows: 1000
Columns: 1
Column name: target
Missing values: 0
Class 0 predictions: 580 (58.0%)
Class 1 predictions: 420 (42.0%)
```

The file is read back and compared with the in-memory submission. If Excel locks the target file, the notebook reports an environmental issue and validates a temporary Excel file instead.

Compared with the previous submission file:

```text
Changed predictions: 81
0 -> 1 changes: 37
1 -> 0 changes: 44
```

## Verified Notebook Status

Latest full execution:

- 39 code cells executed with sequential counts 1 through 39.
- No cell errors.
- 40 Markdown cells.
- All transformed train, validation, and test frames contain 23 features and zero missing values.
- No fitting on validation or final test data.
- No use of `X_valid` inside Decision Tree or MLP hyperparameter tuning.
- Final test row count and order preserved.
- Final Excel export passed all structural checks.
- Raw source CSV hashes remain unchanged.

Environment-only messages observed during independent execution:

- The default MLP emitted a convergence warning at 500 iterations.
- Joblib on Windows emitted resource-tracker cleanup messages after parallel work completed.
- These messages did not produce notebook cell errors or alter the successful export.

## Presentation And Writing Rules

- Markdown explanations and conclusions remain in Hebrew with RTL wrappers.
- Code and code-generated table/plot labels remain in English.
- Conclusions must match saved outputs exactly.
- Explain that Cross-Validation is training-only and the validation set is held out until tuning is complete.
- Do not claim that execution alone proves methodological correctness.
- Keep the notebook suitable for classroom presentation and final report use.

## Git Notes

At the latest check, the working branch was `elad`, tracking `origin/elad`. Always verify with `git status --short --branch` before committing.
