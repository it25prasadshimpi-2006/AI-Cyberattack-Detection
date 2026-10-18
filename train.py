from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

TRAIN_PATH = DATA / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA / "UNSW_NB15_testing-set.csv"

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise SystemExit(
        "\nDataset files are missing.\n"
        "Download UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv "
        "from https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
        f"Place both files in: {DATA}\n"
    )

print("Loading dataset files...")
train = pd.read_csv(TRAIN_PATH)
print("Training columns:", train.columns.tolist())
print("Label exists:", "label" in train.columns)
test = pd.read_csv(TEST_PATH, sep="\t", header=None, names=pd.read_csv(TRAIN_PATH, nrows=0).columns, low_memory=False)
print("Testing columns:", test.columns.tolist())
print("Testing label exists:", "label" in test.columns)

if "label" not in train.columns or "label" not in test.columns:
    raise SystemExit("Expected binary target column 'label' was not found. Check the dataset version.")

y_train = pd.to_numeric(train["label"], errors="raise").astype(int)
y_test = pd.to_numeric(test["label"], errors="raise").astype(int)
if not set(y_train.unique()).issubset({0, 1}) or not set(y_test.unique()).issubset({0, 1}):
    raise SystemExit("Expected binary labels 0=Normal and 1=Attack.")

drop_cols = ["label", "attack_cat", "id"]
X_train = train.drop(columns=drop_cols, errors="ignore").copy()
X_test = test.drop(columns=drop_cols, errors="ignore").copy()
X_test = X_test.reindex(columns=X_train.columns)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Ensure numeric-looking columns are handled as numeric where possible.
for col in X_train.columns:
    if X_train[col].dtype == "object":
        train_converted = pd.to_numeric(X_train[col], errors="coerce")
        # Convert only if almost all non-missing values can be parsed as numeric.
        nonmissing = X_train[col].notna().sum()
        parsed = train_converted.notna().sum()
        if nonmissing and parsed / nonmissing >= 0.98:
            X_train[col] = train_converted
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1200, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
}

rows = []
reports = {}
matrices = {}
best_score = -1.0
best_model = None
best_name = None
fitted_pipelines = {}

for name, classifier in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)
    fitted_pipelines[name] = pipeline
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]

    scores = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba))
    }
    rows.append(scores)
    reports[name] = classification_report(
        y_test, pred, target_names=["Normal", "Attack"],
        output_dict=True, zero_division=0
    )
    matrices[name] = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()

    print(pd.Series(scores).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["Normal", "Attack"], zero_division=0))
    print("Confusion matrix (rows=actual [Normal, Attack], columns=predicted [Normal, Attack]):")
    print(np.array(matrices[name]))

    if scores["f1_score"] > best_score:
        best_score = scores["f1_score"]
        best_model = pipeline
        best_name = name

pd.DataFrame(rows).to_csv(RESULTS / "model_comparison.csv", index=False)
(RESULTS / "classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
(RESULTS / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2), encoding="utf-8")
joblib.dump(best_model, MODELS / "best_model.joblib")
(MODELS / "feature_columns.json").write_text(json.dumps(X_train.columns.tolist(), indent=2), encoding="utf-8")

# Save Random Forest feature importance using transformed feature names.
rf_pipeline = fitted_pipelines.get("Random Forest")
if rf_pipeline is not None:
    try:
        fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
        fitted_rf = rf_pipeline.named_steps["classifier"]
        transformed_names = fitted_preprocessor.get_feature_names_out()
        importance = pd.DataFrame({
            "feature": transformed_names,
            "importance": fitted_rf.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(RESULTS / "feature_importance.csv", index=False)
    except Exception as exc:
        warnings.warn(f"Could not save feature importance: {exc}")

print("\nTraining complete.")
print(f"Selected model by test-set F1-score: {best_name} (F1={best_score:.4f})")
print(f"Metrics saved to: {RESULTS / 'model_comparison.csv'}")
print(f"Model saved to: {MODELS / 'best_model.joblib'}")
print("\nResearch note: use a validation split for model/threshold selection before a final, unbiased test evaluation.")
