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
7. Implemented Decision Tree section.
8. Implemented Neural Network / MLP section.
9. Implemented model comparison section.
10. Implemented selected-model section choosing the optimized Decision Tree.
11. Implemented bonus subgroup-performance analysis on the validation set.
12. Implemented final retraining, prediction, validation, and Excel export section.

The lecturer removed the K-Means and Naive Bayes requirements from the current working scope. The official `part b instructions.md` still includes those older requirements, but the lecturer update overrides them for the working notebook.

The notebook currently implements the Decision Tree, MLP, model comparison, selected-model, bonus-analysis, and final-prediction sections. Section 11 selects the optimized Decision Tree, the bonus analysis examines subgroup behavior on validation data, and Section 12 retrains the tree on all cleaned training data and exports the final predictions.

Do not reintroduce full K-Means or Naive Bayes sections unless explicitly requested. Section 10 now includes only a short conceptual answer explaining why supervised models (`Decision Tree` / `Neural Network`) are preferred over `K-Means` for this classification task, while the numerical comparison remains only between `Decision Tree` and `MLP`.

The previous draft reminders in Sections 11 and 12 have been removed. The old draft texts `shi changed the code below` and `Add blockquote` are also no longer present.

The saved notebook currently contains code-cell outputs from a recent user execution. Independent temporary executions are also used for verification so tests do not overwrite the submitted notebook or final prediction file.

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

Future modeling should encode the target as binary only after the cleaning/splitting setup is clear. Recommended mapping:

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

## Current Decision Tree Section

Section 8, Decision Tree, is implemented.

Important implementation details:

- Uses only `X_train`, `X_valid`, `y_train`, and `y_valid`.
- Does not use `X_test_final` for training, tuning, or validation.
- Encodes the target as:

```text
neutral or dissatisfied -> 0
satisfied -> 1
```

- Uses an sklearn `Pipeline` with a `ColumnTransformer`.
- Numeric features are passed through without scaling.
- Categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")`.
- The code defines helper functions `make_dt_preprocessor()` and `build_dt_pipeline()`.

Full Decision Tree:

```python
DecisionTreeClassifier(random_state=42)
```

Latest verified output for the full tree:

```text
Train Accuracy: 1.0000
Validation Accuracy: 0.8478
Validation Precision: 0.8216
Validation Recall: 0.8321
Validation F1: 0.8268
```

The full tree shows clear overfitting: perfect training performance and lower validation performance.

Decision Tree tuning currently includes:

- Pre-pruning scan over `max_depth` values 1 through 15.
- Cost-complexity pruning using `cost_complexity_pruning_path`.
- One-dimensional sweeps for `min_samples_split` and `min_samples_leaf`.
- A combined 4D scan over `max_depth`, sampled `ccp_alpha`, `min_samples_split`, and `min_samples_leaf`.

The current Decision Tree code uses this expanded 4D search:

```text
depth_options = [6, 8, 9, 10, 12]
min_samples_split_options = [30, 60, 80, 100]
min_samples_leaf_options = [15, 25, 35, 45]
sampled ccp_alpha values from cost-complexity path
```

The code, Hebrew Markdown conclusions, and latest full verification run agree on these results:

```text
Best max_depth from pre-pruning scan: 8
Validation Accuracy at best max_depth: 0.8761

Best ccp_alpha from pruning scan: 0.000507
Validation Accuracy at best ccp_alpha: 0.8883

Expanded combined 4D scan:
Optimal max_depth: 9
Optimal ccp_alpha: 0.0
Optimal min_samples_split: 30
Optimal min_samples_leaf: 25
Validation Accuracy: 0.8917
```

The currently selected best tree is trained dynamically with:

```python
DecisionTreeClassifier(
    max_depth=best_combined_depth,
    ccp_alpha=best_combined_alpha,
    min_samples_split=best_combined_split,
    min_samples_leaf=best_combined_leaf,
    random_state=RANDOM_STATE,
)
```

Latest verified selected-tree metrics:

```text
Train Accuracy: 0.9036
Train Precision: 0.8979
Train Recall: 0.8790
Train F1: 0.8884

Validation Accuracy: 0.8917
Validation Precision: 0.8863
Validation Recall: 0.8626
Validation F1: 0.8743
```

Report-ready Decision Tree outputs currently included:

- Full tree metrics table.
- Full tree train/validation performance plot.
- Generalization gap plot.
- `max_depth` tuning plot.
- Cost-complexity impurity plot.
- Tree complexity plots by `ccp_alpha`.
- Accuracy by `ccp_alpha` plot.
- Combined `max_depth` and `ccp_alpha` scan output.
- Selected pruned tree metrics table.
- Comparison table between full and pruned tree.
- Pruned tree visualization limited to depth 3.
- Validation confusion matrix for the selected tree.
- Manual decision-path example for one validation record.
- Feature importance plot and top-feature table.
- Hebrew conclusions and interpretations.

Current manual decision-path example:

```text
Validation position: 10
Original row index: 4267
True label: Satisfied
Prediction: Satisfied
Correct classification: Yes
```

All code-generated Decision Tree output is in English. Notebook Markdown explanations remain in Hebrew and RTL.

---

## Current MLP / Neural Network Section

Section 9, Neural Network / MLP, is implemented.

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
MLPClassifier(random_state=42, max_iter=500)
```

Current tuning uses:

```python
GridSearchCV(
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True,
    refit=True,
)
```

The MLP tuning search compares predefined architectures and two alpha values. The search MLP uses early stopping to avoid very long runs.

Current search space:

```text
hidden_layer_sizes:
(25,), (50,), (100,), (150,),
(50, 25), (100, 50), (150, 75), (200, 100),
(50, 25, 10), (100, 50, 25)

activation: relu
alpha: 0.0001, 0.001
learning_rate_init: 0.001
```

Current best tuned MLP configuration:

```text
activation: relu
hidden_layer_sizes: (150, 75)
alpha: 0.0001
learning_rate_init: 0.001
```

This means the selected network has two hidden layers, with 150 neurons in the first hidden layer and 75 neurons in the second hidden layer.

Current notebook Markdown records the tuned MLP results as:

```text
Train Accuracy: 0.9261
Validation Accuracy: 0.8883
Generalization gap: about 0.0378
Best mean CV Accuracy: 0.8929
```

The default MLP Markdown records:

```text
Train Accuracy: 0.9933
Train Precision: 0.9946
Train Recall: 0.9901
Train F1: 0.9923

Validation Accuracy: 0.8822
Validation Precision: 0.8747
Validation Recall: 0.8524
Validation F1: 0.8634
```

These MLP values were confirmed in the latest complete verification run. The tuned MLP slightly improves validation accuracy and reduces overfitting compared with the default MLP.

Report-ready MLP outputs currently included:

- Default MLP metrics table.
- MLP architecture summary table.
- Cross-validation tuning results table.
- Tuning plot.
- Tuned MLP metrics table.
- Tuned MLP confusion matrix.
- Hebrew Markdown interpretation plus an English code-generated conclusion comparing default and tuned MLP.

All code-generated display text is intentionally in English, including tables, print messages, automatic conclusions, plot titles, axes, legends, and class labels. Notebook Markdown explanations remain in Hebrew and use RTL/right-aligned wrappers.

---

## Current Model Comparison Section

Section 10, model comparison, is implemented.

Current Section 10 structure:

1. Hebrew Markdown answer explaining why supervised models (`Decision Tree` / `Neural Network`) are preferred over `K-Means` for this task.
2. Hebrew Markdown clarification that, due to the lecturer's updated scope, the numerical comparison continues only between the two remaining supervised models: selected Decision Tree and tuned MLP.
3. A comparison table built from existing variables only:

```python
best_dt_metrics
best_mlp_metrics
best_combined_depth
best_combined_alpha
best_combined_split
best_combined_leaf
best_mlp_params
best_hidden_layers
```

4. A grouped validation metrics plot comparing `Accuracy`, `Precision`, `Recall`, and `F1`.
5. Hebrew conclusion selecting `Decision Tree` as the current leader based on validation behavior, with the formal explanation continued in Section 11.

Important rules for Section 10:

- Do not train new models in Section 10.
- Do not use `X_test_final` in Section 10.
- Do not compare numerically against K-Means or Naive Bayes unless the user explicitly asks to reintroduce those older requirements.
- Keep the conceptual K-Means answer brief and connected to the assignment question: K-Means is unsupervised and does not learn directly from `satisfaction`, while DT/NN are supervised classifiers.
- Keep plot text in English and Markdown explanations in Hebrew.

Latest verified comparison:

```text
Decision Tree Validation Accuracy: 0.8917
Decision Tree Validation F1: 0.8743
Decision Tree Generalization Gap: 0.0119

MLP Validation Accuracy: 0.8883
MLP Validation F1: 0.8714
MLP Generalization Gap: 0.0378
```

The current comparison therefore selects the optimized Decision Tree as the provisional final model.

---

## Current General Bonus Analysis

A bonus section was added between Sections 11 and 12. It analyzes the selected Decision Tree only on the validation set and does not use `X_test_final`.

The section reports subgroup metrics by:

- `Type of Travel`.
- `Customer Type`.

For each group it displays the number of records, number of truly satisfied passengers, Accuracy, Recall for the `Satisfied` class, false negatives, and false positives. A report-ready plot compares Accuracy with Satisfied Recall by travel type.

Main verified finding:

```text
Personal Travel:
Records: 546
Actual Satisfied: 56
Accuracy: 0.9432
Satisfied Recall: 0.4643
False Negatives: 30

Business travel:
Records: 1254
Actual Satisfied: 730
Accuracy: 0.8692
Satisfied Recall: 0.8932
False Negatives: 78
```

The high Accuracy for personal travel hides weak identification of satisfied passengers: only 46.43% were detected correctly. This occurs because the subgroup is strongly dominated by neutral/dissatisfied passengers.

A secondary finding is that Satisfied Recall is `0.6389` for disloyal customers versus `0.8852` for loyal customers. The Hebrew conclusion explains that overall Accuracy should be supplemented with subgroup Recall when evaluating practical model performance.

All bonus tables and plot labels are in English; interpretation remains in Hebrew Markdown and RTL.

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
- Keep code in normal Python.
- Keep all user-visible output generated by code in English: table headers and row labels, print messages, automatic conclusions, validation/error messages, plot titles, axes, legends, and class labels.
- Hebrew may remain in Markdown cells and code comments, but not in code-generated output.
- Avoid long English explanations in Markdown cells.
- Keep the solution clear enough to explain in class.

---

## Current Completion Status

Implemented under the current lecturer-updated scope:

1. Data loading, cleaning, feature engineering, and train/validation split.
2. Decision Tree training, tuning, evaluation, visualization, and interpretation.
3. MLP training, tuning, cross-validation, evaluation, and interpretation.
4. Model comparison and selection of the optimized Decision Tree.
5. Bonus subgroup-performance analysis on the validation set.
6. Full-data retraining of the selected model.
7. Prediction on `X_test_final` with row count and order preserved.
8. Export and validation of the final Excel prediction file.

No additional notebook modeling section remains under the current scope. Final review and submission packaging are still required outside the implementation itself.

Do not add full K-Means or Naive Bayes sections under the current lecturer-updated scope. The old official instructions still mention them, but the current working notebook follows the lecturer update that removed those sections.

The final report should use the verified model-selection, confusion-matrix, full-training, and prediction-export results already produced by the notebook.

---

## Final Prediction File

Section 12 is implemented and creates:

```text
airline_G3_ytest.xlsx
```

Implementation details:

- Maps the full cleaned target with `TARGET_MAP`.
- Builds a new Decision Tree pipeline using `best_combined_depth`, `best_combined_alpha`, `best_combined_split`, and `best_combined_leaf`.
- Retrains on all `8998` cleaned training rows.
- Predicts on all `1000` rows of `X_test_final`.
- Uses `y test example.xlsx` to validate the required single-column structure, not its row count.
- Writes one column named `target` with binary values 0/1.
- Reads the exported Excel file back and compares it with the in-memory submission.

Verified final file:

```text
Rows: 1000
Columns: 1
Column name: target
Missing predictions: 0
Predicted class 0: 573 rows (57.3%)
Predicted class 1: 427 rows (42.7%)
```

The final prediction file:

- Preserve the original order of rows in `X_test.csv`.
- Have the same number of rows as `X_test.csv`.
- Use the same structure as `y test example.xlsx`.
- Use a single output column named:

```text
target
```

Required and generated filename for Group 3 on the airline dataset:

```text
airline_G3_ytest.xlsx
```

The notebook validates:

```python
assert len(submission) == len(test_raw)
assert list(submission.columns) == ["target"]
assert set(submission["target"].unique()).issubset({0, 1})
```

---

## Latest Verification Status

The notebook was executed completely after implementing Section 12, and a second temporary execution through the newly added bonus section verified the subgroup analysis.

Verified:

- Full execution completed successfully.
- The notebook contains 40 Markdown cells and 39 code cells.
- Preprocessing logic, model logic, data split, hyperparameters, and metrics remained unchanged.
- No Hebrew appeared in executed code outputs.
- Training rows after cleaning: `8998`.
- Split sizes: `7198` training rows and `1800` validation rows.
- Final test rows: `1000`, with row count and order preserved.
- Optimized Decision Tree validation Accuracy: `0.8917`.
- Tuned MLP validation Accuracy: `0.8883`.
- Bonus travel-type and customer-type tables matched the documented subgroup findings.
- Bonus code-generated output contains no Hebrew; the plot labels are in English.
- Section 12 retrained the optimized Decision Tree on all `8998` cleaned training rows.
- `airline_G3_ytest.xlsx` was created successfully with `1000` rows and one `target` column.
- Final test row count and order were preserved.
- The exported Excel file matched the in-memory submission exactly.

The temporary execution result was used only for verification. The saved notebook retains its existing stored code-cell outputs.

---

## Git / Branch Notes

At the time of this context update, the repository is on branch `main`, tracking `origin/main`. Always run `git status -sb` before committing or pushing.

The local reference notebook:

```text
Airline_Project_Part_B_Group3 (3).ipynb
```

was used earlier as a local reference for corrected Part A preprocessing and is not part of the final single-notebook submission. It may not be present on every branch or clone.
