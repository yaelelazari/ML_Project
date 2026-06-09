# CONTEXT.md - Machine Learning Project Part B

## Project Identity

This repository is for **Machine Learning 364-1-1811, Project Part B**, Group 3, using the **Airline Satisfaction** dataset.

The final submitted code artifact should be one notebook:

```text
Airline_Project_Part_B_Group3.ipynb
```

The notebook should run from top to bottom using the raw project files. It should include the corrected Part A cleaning logic internally, then continue to Part B modeling, comparison, final model selection, and final prediction export.

The reference notebook:

```text
Airline_Project_Part_B_Group3 (3).ipynb
```

is only a local/reference source for the corrected Part A preprocessing decisions. It is not the final notebook.

---

## Current Required Input Files

The final notebook should assume these raw files are available in the same folder:

```text
Xy_train.csv
X_test.csv
y test example.xlsx
AirlineSatisfaction Dataset.docx
```

Use:

- `Xy_train.csv` for training and internal validation.
- `X_test.csv` only for final predictions.
- `y test example.xlsx` only to confirm the required prediction-file structure.

Do not require intermediate cleaned CSV files such as `Xy_train_clean.csv` or `X_test_clean.csv`. The cleaning is done inside the notebook and kept in memory.

---

## Current Notebook State

`Airline_Project_Part_B_Group3.ipynb` currently contains:

1. Imports and constants.
2. Loading of raw train, final test, and example submission files.
3. Basic schema checks.
4. Automatic target-column detection from the raw training file.
5. Corrected Part A cleaning/preprocessing integrated directly into the notebook.
6. Train/validation split.
7. Placeholder Markdown section for Decision Tree.
8. Implemented Neural Network / MLP section.
9. Placeholder sections for K-Means, Naive Bayes, model comparison, final model, and final prediction export.

The notebook currently implements only the MLP model section. It does **not** yet implement:

- Decision Tree
- K-Means
- Naive Bayes
- Model comparison
- Final model
- Final prediction export

Do not implement those remaining model sections unless explicitly requested.

---

## Target Variable

The target column is detected from the loaded files instead of hard-coding a name. In the current dataset it is:

```text
satisfaction
```

Known target values:

```text
satisfied
neutral or dissatisfied
```

The MLP section encodes the target as binary with this mapping:

```python
TARGET_MAP = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}
```

Never impute the target column. Rows with missing target cannot be used for supervised learning and must be removed from the training data only.

---

## Corrected Part A Cleaning Logic Currently Integrated

The current final notebook applies these cleaning decisions internally.

### Class

- Valid values are:

```text
Eco
Eco Plus
Business
Unknown
```

- Missing `Class` values are replaced with `Business`.
- `Class = Unknown` is replaced with `Business`.
- Invalid non-missing free-text `Class` values are removed from training data only.
- Final test rows are never removed. If an invalid `Class` value appears in final test, it is treated as missing and then replaced with `Business`.

Important current row behavior in `Xy_train.csv`:

- Row index `5568`: missing `Class`; kept and changed to `Business`.
- Row index `8732`: free-text `Class`; removed from training.
- Row index `5602`: missing target; removed from training.

### Missing Target

- Rows with missing target are removed from training before general categorical imputation.
- This is done only for the training data.
- The final test file has no target and must never have rows removed.
- The target column is excluded from all numeric and categorical imputation.

### Gate Location

- Invalid `Gate location` values are converted to `NaN`.
- They are then imputed using the median, matching the corrected Part A notebook logic.

### Leg Room Service

Missing `Leg room service` is imputed hierarchically:

1. Median by `Class` + `Type of Travel`.
2. Median by `Class`.
3. Global median.

### Age

- Invalid `Age` values, such as values below 0 or above 110, are converted to `NaN`.
- They are then imputed using the median.

### Flight Distance

- Negative `Flight Distance` values are converted to `NaN`.
- They are then imputed hierarchically:

1. Median by `Class` + `Type of Travel`.
2. Median by `Class`.
3. Global median.

### Plane Colors

- `Plane colors` is dropped.
- The drop must happen from the currently cleaned dataframe.
- Never reset the cleaned dataframe from the raw dataframe.

Correct pattern:

```python
if "Plane colors" in df_clean.columns:
    df_clean = df_clean.drop(columns=["Plane colors"])
```

Do not use:

```python
df_clean = df.drop(columns=["Plane colors"])
```

### Remaining Missing Values

After the specific cleaning steps:

- Remaining numeric missing values are imputed with median.
- Remaining categorical missing values are imputed with mode.
- The target column is excluded from these imputations.

### Engineered Features

The notebook creates:

```text
Age_Category
Flight_Type
Total_Service_Score
```

`Total_Service_Score` is calculated as the mean of the service-rating columns.

---

## Current Verified Cleaning Output

After running the current notebook:

```text
Raw training rows: 9000
Training rows after cleaning: 8998
Final test rows before cleaning: 1000
Final test rows after cleaning: 1000
```

Rows removed from training:

- 1 row with invalid free-text `Class`.
- 1 row with missing target.

Rows not removed:

- The row with missing `Class` is kept and `Class` is replaced with `Business`.

Current checks pass:

```text
X_test row count preserved: True
X_test row order preserved: True
Missing values in X_model: 0
Missing values in y_model: 0
Missing values in X_test_final: 0
Plane colors removed from train and test: True
Engineered features created: True
```

After cleaning:

```text
X_model shape: 8998 rows, 23 features
X_test_final shape: 1000 rows, 23 features
```

After the current 80/20 stratified split:

```text
X_train rows: 7198
X_valid rows: 1800
X_test_final rows: 1000
```

---

## Current MLP / Neural Network Section

Section 9, `????? ???????? / MLP`, is implemented.

Important implementation details:

- Uses only `X_train`, `X_valid`, `y_train`, and `y_valid`.
- Does not use `X_test_final` for training, tuning, or validation.
- Encodes the target as:

```text
neutral or dissatisfied -> 0
satisfied -> 1
```

- Uses an sklearn `Pipeline` so preprocessing is fitted only on the training data during training and cross-validation.
- Numeric features are scaled with `StandardScaler`.
- Categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")`.
- The preprocessed MLP input size is currently 33 features.

Default MLP:

```python
MLPClassifier(random_state=42, max_iter=1000)
```

Compact tuning uses:

```python
RandomizedSearchCV(
    cv=3,
    scoring="accuracy",
    n_iter=12,
    random_state=42,
)
```

The tuned MLP search is intentionally compact for Colab runtime. The search MLP uses early stopping to avoid very long runs.

Current best tuned MLP configuration:

```text
activation: relu
hidden_layer_sizes: (100,)
alpha: 0.01
learning_rate_init: 0.01
```

This means the selected network has one hidden layer with 100 neurons.

Current validation results for the tuned MLP:

```text
Accuracy: 0.8906
Precision: 0.8780
Recall: 0.8702
F1: 0.8741
```

The default MLP had very high training performance and lower validation performance, so the tuned MLP generalizes better.

Report-ready MLP outputs currently included:

- Default MLP metrics table.
- MLP architecture summary table.
- Cross-validation tuning results table.
- Tuning plot.
- Tuned MLP metrics table.
- Tuned MLP confusion matrix.
- Short Hebrew conclusions comparing default and tuned MLP.

Plot display text is intentionally in English because Hebrew RTL rendering in Matplotlib is unreliable. Notebook Markdown explanations remain in Hebrew.

---

## Train/Validation Split

Part B requires splitting `Xy_train.csv` into training and validation sets.

Current split approach:

```python
train_test_split(
    X_model,
    y_model,
    test_size=0.2,
    random_state=42,
    stratify=y_model,
)
```

Rationale:

- 80/20 keeps enough data for training while preserving a meaningful validation set.
- Stratification preserves the class proportions.
- The validation set simulates future unseen observations.
- `X_test.csv` remains separate and is used only for final prediction.

---

## Important Workflow Rules

- Submit one final notebook: `Airline_Project_Part_B_Group3.ipynb`.
- Raw files remain unchanged.
- Do not require cleaned intermediate CSV files.
- Do not use `X_test.csv` for model selection, tuning, or validation.
- Never remove rows from final test.
- Never change final test row order.
- Never impute the target.
- Keep all Markdown explanations, interpretations, and conclusions in Hebrew.
- Markdown cells in the notebook are wrapped with RTL/right-align display markup.
- Keep code in normal Python.
- Plot titles, axis labels, legends, and class labels inside Matplotlib plots should be in English for readability.
- Avoid long English explanations in Markdown cells.
- Keep the solution clear enough to explain in class.

---

## Required Part B Modeling Sections To Add Later

Remaining model sections to add with Hebrew explanations:

1. Decision Tree
2. K-Means
3. Naive Bayes
4. Model comparison
5. Final selected model
6. Final prediction export

The Neural Network / MLP section is already implemented.

Required outputs for the report should include:

- Metrics tables.
- Hyperparameter tuning summaries.
- Confusion matrices.
- Tree visualization and feature importance for Decision Tree.
- K-Means cluster-class comparison and visualization.
- Naive Bayes predicted probabilities for one validation record.
- Model comparison table.
- Final submission Excel file.

---

## Final Prediction File

The final prediction file must:

- Preserve the original order of rows in `X_test.csv`.
- Have the same number of rows as `X_test.csv`.
- Use the same structure as `y test example.xlsx`.
- Use a single output column named:

```text
target
```

Required filename for Group 3 on the airline dataset:

```text
airline_G3_ytest.xlsx
```

Before export, validate:

```python
assert len(submission) == len(test_raw)
assert list(submission.columns) == ["target"]
assert set(submission["target"].unique()).issubset({0, 1})
```

---

## Git / Branch Notes

The current working branch is intended for ongoing work. Recent changes were pushed to `main`, and a local branch named `elad` was created for separate work.

The local reference notebook:

```text
Airline_Project_Part_B_Group3 (3).ipynb
```

has intentionally not been pushed as part of the final notebook workflow unless explicitly requested.
