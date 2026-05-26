# CONTEXT.md — Machine Learning Project Part B

## Project identity

This repository is for **Machine Learning 364-1-1811, Project Part B**, Group 3, using the **Airline Satisfaction** dataset.

The goal of Part B is to continue from the data understanding and preprocessing work done in Part A and build, tune, compare, and submit machine-learning models for predicting passenger satisfaction.

Work carefully and transparently. The course explicitly warns that code or explanations that look generated, ungrounded, or not understood may reduce the grade. Every modeling decision must be explainable in simple Hebrew in the final report.

---

## Main deliverables

At the end of Part B, the project must produce exactly these deliverables:

1. **Word report** for Part B.
2. **Python code file / notebook** with all preprocessing, training, validation, tuning, comparison, and final prediction steps.
3. **Excel prediction file** for `X_test.csv`, following the same structure as `y test example.xlsx`.

The final prediction file must:

- Preserve the original order of rows in `X_test.csv`.
- Have the same number of rows as `X_test.csv`.
- Use the same output structure as `y test example.xlsx`, whose column is `target`.
- Be named according to the required format. For Group 3 on Airline data, use:

```text
airline_G3_ytest.xlsx
```

---

## Available files in the project folder

Expected files:

```text
Xy_train.csv
X_test.csv
y test example.xlsx
AirlineSatisfaction Dataset.docx
Airline_Project_Part_B_Group3 (3).ipynb
הוראות חלק א.docx
הוראות חלק ב.docx
```

Use `Xy_train.csv` for training and internal validation.
Use `X_test.csv` only for final predictions.
Do not use `X_test.csv` during model selection except to confirm schema and preprocessing compatibility.

---

## Dataset description

The dataset contains an airline passenger satisfaction survey. The research question is:

> Can passenger satisfaction be predicted from passenger profile, travel characteristics, service ratings, and delay information?

### Target variable

In the CSV file, the target column is named:

```text
satisfaction
```

The target values are:

```text
satisfied
neutral or dissatisfied
```

For modeling and final submission, encode the target as binary.
Recommended mapping:

```python
TARGET_MAP = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}
```

The training file has approximately:

```text
9000 rows, 22 columns
```

The final test file has approximately:

```text
1000 rows, 21 columns
```

The target distribution in `Xy_train.csv` is moderately imbalanced but not extreme:

```text
neutral or dissatisfied: about 56.35%
satisfied: about 43.65%
```

Because the imbalance is mild, accuracy may be acceptable as a main metric, but also report at least F1-score, recall, precision, and confusion matrix for the selected model. If optimizing for the competition, prefer a metric that reflects validation performance robustly, not only training accuracy.

---

## Column meanings

Based on `AirlineSatisfaction Dataset.docx`, the columns are:

| Column | Meaning | Suggested type |
|---|---|---|
| `Gender` | Passenger gender: Female/Male | Nominal categorical |
| `Customer Type` | Loyal or disloyal customer | Nominal categorical |
| `Age` | Passenger age | Continuous/numeric |
| `Type of Travel` | Personal or business travel | Nominal categorical |
| `Class` | Business, Eco, Eco Plus | Nominal/ordinal-ish categorical |
| `Flight Distance` | Journey distance | Continuous/numeric |
| `Plane colors` | Number of colors on plane body | Suspicious/noisy numeric/categorical |
| `Inflight wifi service` | Satisfaction rating, 0 not applicable, 1–5 | Ordinal numeric |
| `Departure/Arrival time convenient` | Satisfaction rating | Ordinal numeric |
| `Ease of Online booking` | Satisfaction rating | Ordinal numeric |
| `Gate location` | Satisfaction rating | Ordinal numeric |
| `Food and drink` | Satisfaction rating | Ordinal numeric |
| `Seat comfort` | Satisfaction rating | Ordinal numeric |
| `On-board service` | Satisfaction rating | Ordinal numeric |
| `Leg room service` | Satisfaction rating | Ordinal numeric |
| `Baggage handling` | Satisfaction rating | Ordinal numeric |
| `Checkin service` | Satisfaction rating | Ordinal numeric |
| `Inflight service` | Satisfaction rating | Ordinal numeric |
| `Cleanliness` | Satisfaction rating | Ordinal numeric |
| `Departure Delay in Minutes` | Departure delay | Continuous/numeric |
| `Arrival Delay in Minutes` | Arrival delay | Continuous/numeric |
| `satisfaction` | Target class | Binary categorical |

---

## Current notebook status and important warning

The uploaded notebook `Airline_Project_Part_B_Group3 (3).ipynb` currently contains mainly corrected preprocessing after feedback on Part A. It is not yet a complete Part B modeling notebook.

Important preprocessing already discussed in the notebook:

1. Remove one row with invalid free-text value in `Class`:

```text
IT IS SO BORING WORKING IN AN AIRPORT'S DESK OH MY GODDDDD
```

2. Replace `Class = Unknown` with `Business`, based on its relationship with `Type of Travel`.
3. Treat invalid `Gate location = 999` by converting only that cell to missing and imputing, rather than deleting the full row.
4. Treat biologically impossible `Age` values, such as values above 110, by converting only the invalid cell to missing and imputing.
5. Treat negative `Flight Distance` values by converting only the invalid cell to missing and imputing.
6. Impute `Leg room service`, which has many missing values, using a hierarchical strategy:
   - first by `Class` + `Type of Travel`,
   - then by `Class`,
   - then by global median.
7. Consider dropping `Plane colors` as a noisy/irrelevant feature, but keep the option to test both with and without it.
8. Add possible engineered features:
   - `Age_Category`
   - `Flight_Type`
   - `Total_Service_Score`

### Critical bug to fix before continuing

There is a serious bug in the current notebook around dropping `Plane colors`:

```python
# WRONG: this resets df_clean from the original df and loses earlier cleaning steps
# df_clean = df.drop(columns=['Plane colors'])
```

It must be changed to:

```python
# Correct: continue from the already cleaned dataframe
if "Plane colors" in df_clean.columns:
    df_clean = df_clean.drop(columns=["Plane colors"])
```

Never reset `df_clean` from `df` after cleaning has started.

---

## General preprocessing principles

Create a reusable preprocessing pipeline, not scattered one-off transformations.

The same preprocessing logic must be applied to:

1. training split,
2. validation split,
3. cross-validation folds,
4. final `X_test.csv`.

For final `X_test.csv`, rows must never be deleted. If problematic values appear there, convert problematic cells to missing and impute.

Recommended structure:

```text
load data
clean target and split X/y
train-validation split with stratify=y
fit preprocessing only on training data
transform validation data
train/tune models
choose final model
fit final pipeline on all cleaned Xy_train
transform X_test using the same pipeline
predict
export Excel
```

Avoid data leakage. Do not fit scalers, imputers, encoders, PCA, or feature selectors on the full dataset before validation split or cross-validation.

---

## Train/validation split

Part B requires splitting `Xy_train.csv` into training and validation sets.

Recommended split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

Rationale for report:

- 80/20 keeps enough data for training while preserving a meaningful validation set.
- Stratification preserves the original class proportions.
- The validation set simulates future unseen data and is separate from `X_test.csv`, which is only for final submission.

---

## Model-specific preprocessing

### Decision Tree

Decision trees do not require feature scaling.

Required preprocessing:

- Impute missing numeric values.
- Impute missing categorical values.
- Encode categorical variables, preferably One-Hot Encoding.
- Optionally include engineered features.

Use `DecisionTreeClassifier` inside an `sklearn.pipeline.Pipeline` with `ColumnTransformer`.

Important course concepts to reflect in report:

- A decision tree is readable as if-then rules.
- Splits are chosen to reduce impurity/uncertainty.
- ID3 uses information gain based on entropy, while sklearn commonly uses Gini or entropy/log-loss.
- A full tree can overfit by perfectly or almost perfectly fitting the training data.
- Hyperparameter tuning controls tree complexity and generalization.

Recommended hyperparameters to tune:

```python
param_grid_dt = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [3, 5, 7, 10, 15, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 5, 10],
    "model__max_features": [None, "sqrt", "log2"],
}
```

Report explanations:

- `max_depth`: lower values simplify the tree; higher values allow complex rules and possible overfitting.
- `min_samples_split`: higher values prevent tiny splits.
- `min_samples_leaf`: higher values smooth the tree and reduce sensitivity to outliers.
- `criterion`: measures split quality.
- `max_features`: controls how many features are considered at each split.

Required outputs:

- Full tree training and validation score.
- Tuned tree training and validation score.
- Hyperparameter tuning table/plot.
- Tree visualization, limited depth if needed.
- Manual walk-through of one validation record through the displayed tree.
- Feature importances and explanation.
- Confusion matrix if selected/final.

---

## Neural Network / MLP

Use `sklearn.neural_network.MLPClassifier` unless there is a clear reason to use another implementation.

Required preprocessing:

- Impute missing numeric values.
- Impute missing categorical values.
- One-Hot Encode categorical variables.
- Scale numeric features with `StandardScaler`.

Important course concepts to reflect in report:

- Neural networks are discriminant functions with adaptive weights.
- Hidden layers and nonlinear activation allow the model to learn nonlinear boundaries.
- Logistic sigmoid can be interpreted probabilistically, while ReLU is often efficient in practice.
- Too many hidden units/layers can cause overfitting and long training time.
- Scaling is important because gradient-based learning is sensitive to feature magnitudes.

Default model:

```python
MLPClassifier(random_state=42, max_iter=1000)
```

Explain the default architecture:

- Input layer size equals the number of features after preprocessing and encoding.
- Default hidden layer in sklearn is usually one hidden layer with 100 neurons.
- Output is binary classification.

Recommended compact hyperparameter search:

```python
param_grid_mlp = {
    "model__hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": [0.0001, 0.001, 0.01],
    "model__learning_rate_init": [0.001, 0.01],
}
```

Use a compact search to avoid runs that take too long. Prefer `RandomizedSearchCV` if runtime is high.

Required outputs:

- Default MLP train and validation metrics.
- Tuned MLP train and validation metrics.
- Explanation of selected architecture.
- Hyperparameter tuning table/plot.
- Optional but recommended: confusion matrix.

---

## K-Means

K-Means is unsupervised, so it does not use the target during clustering. However, Part B asks to compare clusters to known classes afterward.

Required preprocessing:

- Use only feature columns, not target.
- Impute missing values.
- One-Hot Encode categorical features.
- Scale features.
- Consider PCA for visualization.

Required first run:

- Set `n_clusters` according to the known number of target classes.
- Since the target has two classes, use:

```python
KMeans(n_clusters=2, random_state=42, n_init=10)
```

Discuss:

- How observations are assigned to clusters: nearest centroid, usually by Euclidean distance.
- Whether clusters align with `satisfaction`.
- Use crosstab between cluster labels and true labels.
- Use PCA 2D plot for visualization.

Optional/extended requirement from instructions:

- Train eight K-Means models with different `K` values.
- Choose K using elbow method, silhouette score, or both.
- Discuss whether the selected K matches the business story. It might not, because unsupervised structure can reflect passenger segments rather than satisfaction classes.
- Compare to another clustering method, such as Agglomerative Clustering or Gaussian Mixture Model, if implementing the optional section.

Recommended metrics:

```text
inertia
silhouette_score
adjusted_rand_score against true target, for diagnostic purposes only
cluster-class crosstab
```

---

## Naive Bayes

Part B requires two Naive Bayes classifiers of different types.

Important course concepts to reflect in report:

- Naive Bayes uses Bayes' theorem.
- It assumes features are conditionally independent given the class.
- This assumption is probably not fully true in this dataset because many service rating variables are correlated.
- Despite the unrealistic assumption, Naive Bayes can still be useful as a simple baseline.

Recommended models:

1. `GaussianNB` for continuous/scaled numeric-like features.
2. `CategoricalNB` or `BernoulliNB` after categorical/ordinal encoding.

Simpler robust implementation:

- Use `GaussianNB` on a fully numeric preprocessed matrix.
- Use `BernoulliNB` after One-Hot Encoding categorical variables and optionally binarizing numeric service scores, if practical.

Required outputs:

- Explanation of independence assumption.
- Explain whether assumption is plausible in this dataset.
- Train and validation performance for first NB model.
- Pick one validation record and show predicted probabilities for both classes.
- Train and validation performance for second NB model.
- Compare both NB models.

---

## Model comparison

The report must compare:

1. selected/tuned Decision Tree,
2. selected/tuned MLP,
3. selected Naive Bayes classifier.

Also explain why DT/NN are preferred over K-Means for classification:

- DT and NN are supervised models trained directly on labeled examples.
- K-Means is unsupervised and only finds geometric clusters; it does not optimize prediction of satisfaction.
- K-Means may uncover passenger segments, but its cluster labels do not necessarily match satisfaction labels.

Comparison table should include:

```text
model
main preprocessing
best hyperparameters
train accuracy
validation accuracy
validation precision
validation recall
validation F1
main strengths
main weaknesses
```

Choose the final model based on validation performance, stability, interpretability, and suitability for final prediction.

---

## Final model and final predictions

After choosing the final model:

1. Train the final pipeline on all available `Xy_train.csv` after cleaning.
2. Apply the identical preprocessing pipeline to `X_test.csv`.
3. Predict binary labels.
4. Export an Excel file with one column named `target`.

Example final export:

```python
submission = pd.DataFrame({"target": final_predictions})
submission.to_excel("airline_G3_ytest.xlsx", index=False)
```

Before export, validate:

```python
assert len(submission) == len(X_test)
assert list(submission.columns) == ["target"]
assert set(submission["target"].unique()).issubset({0, 1})
```

---

## Report writing expectations

The report should be concise and insight-oriented.

For every important code step, explain in Hebrew:

1. Why was it done?
2. What was the output?
3. What did we learn from it?
4. How does it affect the next stage?

Avoid dumping raw outputs. Use selected tables and clear plots.

Important report constraints:

```text
maximum 11 pages, excluding cover, table of contents, appendices, and bonus section
font Arial, size 12
line spacing 1.5
central outputs must appear in the report body, not hidden only in appendices
```

---

## Recommended notebook/code architecture

Build the notebook in this order:

```text
1. Imports and constants
2. Load files
3. Basic schema checks
4. Clean target and define X/y
5. Clean known invalid values
6. Train-validation split
7. Define reusable preprocessing pipelines
8. Decision Tree section
   8.1 full tree
   8.2 hyperparameter tuning
   8.3 best tree analysis
9. MLP section
   9.1 default MLP
   9.2 hyperparameter tuning
   9.3 best MLP analysis
10. K-Means section
11. Naive Bayes section
12. Model comparison
13. Final model training on full train data
14. Final prediction on X_test
15. Export submission Excel
```

Use functions to avoid duplication:

```python
def clean_known_data_issues(df, is_train=True):
    ...

def build_preprocessor(scale_numeric=False):
    ...

def evaluate_classifier(model, X_train, y_train, X_val, y_val):
    ...

def export_submission(predictions, path="airline_G3_ytest.xlsx"):
    ...
```

---

## Coding standards for Codex

When modifying code:

- Prefer clear, simple sklearn pipelines over clever code.
- Keep random seeds fixed with `random_state=42` where possible.
- Do not silently change target mapping.
- Do not change row order in `X_test.csv`.
- Do not delete rows from `X_test.csv`.
- Avoid fitting preprocessors on validation or test data.
- Add comments that explain why, not just what.
- Store important intermediate results in named variables.
- Make plots readable: titles, axes labels, legends, and compact size.
- Keep code and report aligned. If code does something, the report should not say the opposite.

---

## Known inconsistencies to clean before final submission

1. The notebook text says this is Part B, but current content is mostly preprocessing. Add full modeling sections.
2. The notebook has a bug where `df_clean` is reset from `df` when dropping `Plane colors`. Fix this first.
3. There is a markdown note saying the report said no age outliers but the notebook found outliers. The final report and notebook must agree.
4. Remove informal wording from final notebook/report, such as:

```text
זה הדרדור (?)
```

5. Standardize column naming. The dataset description says `Satisfaction`, but the CSV uses lowercase `satisfaction`. Use the CSV column name in code.
6. `Checkin service` appears without a hyphen in the CSV. Do not accidentally rename it to `Check-in service` unless done consistently.

---

## Bonus opportunities

Possible bonus directions, only after the required sections work:

1. Compare final selected model to an additional stronger model, such as Random Forest, Gradient Boosting, XGBoost, LightGBM, or Logistic Regression.
2. Use permutation importance or SHAP-style interpretation for the final model.
3. Show calibration or predicted probability distribution.
4. Try model ensemble if validation improves:
   - soft voting between tuned tree/forest/MLP/NB,
   - or simple averaging of predicted probabilities.
5. Add a nontrivial business insight, for example identifying which service dimensions most separate satisfied from dissatisfied passengers.

Only include bonus work if it is clean, tested, and does not distract from the required sections.

---

## Mental model for the project

Treat Part B as a small machine-learning experiment, not just a coding task.

The narrative should be:

```text
We cleaned and prepared the data from Part A.
We split the data to simulate future unseen observations.
We trained several models with different learning assumptions.
We tuned each model to reduce overfitting and improve generalization.
We compared the models fairly on the same validation set.
We selected the strongest and most reliable model.
We applied the same preprocessing to the hidden test set.
We exported predictions in the required format.
```

The final answer should make the reader feel that every step was controlled, justified, and connected to the course concepts.
