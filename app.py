from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.json"
METRICS_PATH = ROOT / "results" / "model_comparison.csv"
REPORT_PATH = ROOT / "results" / "classification_reports.json"
CM_PATH = ROOT / "results" / "confusion_matrices.json"
IMPORTANCE_PATH = ROOT / "results" / "feature_importance.csv"

st.set_page_config(page_title="AI Cyberattack Detection", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return model, features

model, features = load_artifacts()

st.title("🛡️ AI-Based Cyberattack Detection")
st.caption("Machine-learning prototype for classifying labeled network-flow records")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Open page", ["Overview", "Traffic Prediction", "Model Performance", "About"])

if model is None:
    st.warning("No trained model found yet. Open the project folder in VS Code and run `setup_and_run.bat`, or follow README.md.")
    st.stop()

if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", model.named_steps["classifier"].__class__.__name__)
    c2.metric("Input features", len(features))
    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    c3.metric("Models evaluated", len(metrics))
    st.markdown("""
    This project demonstrates a complete defensive ML workflow:
    - preprocessing of network-flow data
    - training and comparison of classification algorithms
    - evaluation with precision, recall, F1-score and confusion matrices
    - batch predictions through a web dashboard

    **Scope:** this is an educational batch-analysis tool. It does not capture live network traffic or replace a production security system.
    """)
    if METRICS_PATH.exists():
        st.subheader("Model comparison")
        st.dataframe(metrics, use_container_width=True)
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in metrics.columns]
        if metric_cols:
            chart = metrics.melt(id_vars="model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            st.plotly_chart(px.bar(chart, x="Metric", y="Score", color="model", barmode="group", title="Evaluation metrics"), use_container_width=True)

elif page == "Traffic Prediction":
    st.subheader("Predict traffic records")
    st.write("Upload a CSV containing the same input features used during training. The app aligns columns to the trained schema.")
    st.info("For the official UNSW-NB15 test CSV, the app automatically ignores `label`, `attack_cat`, and `id` columns. Predictions are not a substitute for ground-truth evaluation.")
    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            if incoming.empty:
                st.error("The uploaded CSV has no rows.")
                st.stop()
            actual_labels = incoming["label"].copy() if "label" in incoming.columns else None
            data = incoming.drop(columns=["label", "attack_cat", "id"], errors="ignore")
            missing = [c for c in features if c not in data.columns]
            extra = [c for c in data.columns if c not in features]
            if missing:
                st.warning(f"{len(missing)} expected feature(s) are missing. They will be treated as missing values by the model pipeline.")
            if extra:
                st.caption(f"Ignoring {len(extra)} extra column(s) not used by the model.")
            X = data.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
            pred = model.predict(X)
            out = incoming.copy()
            out["predicted_class"] = np.where(np.asarray(pred) == 1, "Attack", "Normal")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                classes = list(model.classes_)
                if 1 in classes:
                    out["attack_probability_percent"] = (probabilities[:, classes.index(1)] * 100).round(2)
            total = len(out)
            attacks = int((np.asarray(pred) == 1).sum())
            normal = total - attacks
            a, b, c = st.columns(3)
            a.metric("Records analyzed", f"{total:,}")
            b.metric("Predicted attacks", f"{attacks:,}")
            c.metric("Predicted attack rate", f"{attacks / total:.1%}" if total else "0%")
            counts = out["predicted_class"].value_counts().rename_axis("Class").reset_index(name="Records")
            st.plotly_chart(px.bar(counts, x="Class", y="Records", color="Class", title="Predicted traffic distribution"), use_container_width=True)
            if actual_labels is not None:
                st.caption("Ground-truth `label` was present in the upload, but it was excluded from model input.")
                try:
                    truth = actual_labels.astype(int).to_numpy()
                    from sklearn.metrics import classification_report, confusion_matrix
                    st.subheader("Optional evaluation against supplied labels")
                    st.dataframe(pd.DataFrame(classification_report(truth, pred, output_dict=True, zero_division=0)).transpose(), use_container_width=True)
                    cm = confusion_matrix(truth, pred, labels=[0, 1])
                    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted Normal", "Predicted Attack"], y=["Actual Normal", "Actual Attack"], labels={"x":"Prediction", "y":"Actual"}, title="Confusion matrix"), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not calculate optional metrics from the supplied labels: {e}")
            st.subheader("Prediction results")
            st.dataframe(out.head(500), use_container_width=True)
            st.download_button("Download all predictions as CSV", out.to_csv(index=False).encode("utf-8"), file_name="cyberattack_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process this file: {e}")
            st.caption("Check that the CSV is valid and contains compatible network-flow features.")

elif page == "Model Performance":
    st.subheader("Model evaluation")
    if not METRICS_PATH.exists():
        st.info("No evaluation results found. Run model training first.")
    else:
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics, use_container_width=True)
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in metrics.columns]
        if metric_cols:
            chart = metrics.melt(id_vars="model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            st.plotly_chart(px.bar(chart, x="Metric", y="Score", color="model", barmode="group", title="Model comparison"), use_container_width=True)
    if CM_PATH.exists():
        matrices = json.loads(CM_PATH.read_text(encoding="utf-8"))
        for name, matrix in matrices.items():
            st.markdown(f"**Confusion matrix — {name}**")
            st.plotly_chart(px.imshow(matrix, text_auto=True, x=["Predicted Normal", "Predicted Attack"], y=["Actual Normal", "Actual Attack"], labels={"x":"Prediction", "y":"Actual"}, title=name), use_container_width=True)
    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH).head(20).sort_values("importance")
        st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Top feature importances (Random Forest)"), use_container_width=True)
    if REPORT_PATH.exists():
        st.download_button("Download classification reports (JSON)", REPORT_PATH.read_bytes(), file_name="classification_reports.json", mime="application/json")

else:
    st.subheader("About this project")
    st.markdown("""
    **Dataset:** UNSW-NB15 (download separately from the official UNSW research page).

    **Algorithms:** Logistic Regression and Random Forest.

    **Metrics:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix.

    **Important limitations**
    - A model trained on this dataset may not generalize to current network traffic.
    - Predictions can produce false positives and false negatives.
    - The app analyzes uploaded CSV files only; it does not sniff packets or monitor live networks.
    - Feature importance indicates model reliance, not causation.

    **Responsible use:** use this project for coursework and defensive security analysis on data you are authorized to analyze.
    """)
    st.markdown("[Official UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)")
