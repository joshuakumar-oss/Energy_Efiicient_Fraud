import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from utils.evaluation import get_confusion_matrix
from utils.pipeline import (
    add_time_features,
    carbon_footprint,
    get_carbon_data,
    prepare_dataset_pair,
    predict_two_stage,
    sample_dataset,
    sustainability_score,
    train_two_stage_model,
)
from utils.plots import plot_confusion_matrix


REQUIRED_TARGET = "is_fraud"

st.set_page_config(page_title="Sentinel AI Fraud Command Center", page_icon=":shield:", layout="wide")


def styles():
    theme = st.session_state.get("theme_mode", "Light")
    is_dark = theme == "Dark"
    bg_end = "#10151f" if is_dark else "#f5efe4"
    bg_start = "#161c28" if is_dark else "#fdf8f0"
    text = "#f3f4f6" if is_dark else "#14213d"
    card = "rgba(24, 31, 44, 0.92)" if is_dark else "rgba(255,249,240,.9)"
    card_border = "rgba(255,255,255,.08)" if is_dark else "rgba(20,33,61,.1)"
    metric_bg = "rgba(17, 24, 39, 0.72)" if is_dark else "rgba(255,255,255,.62)"
    sidebar_text = "#f7f1e8"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Instrument+Sans:wght@400;500;600;700&display=swap');
        .stApp{{background:radial-gradient(circle at top left,rgba(239,131,84,.18),transparent 26%),radial-gradient(circle at top right,rgba(42,157,143,.16),transparent 24%),linear-gradient(180deg,{bg_start} 0%,{bg_end} 100%);font-family:'Instrument Sans',sans-serif;color:{text}}}
        [data-testid="stSidebar"]{{background:linear-gradient(180deg,#18314f 0%,#264653 100%)}}
        [data-testid="stSidebar"] *{{color:{sidebar_text}}}
        .block-container{{max-width:1320px;padding-top:1.4rem;padding-bottom:3rem}}
        h1,h2,h3{{font-family:'Space Grotesk',sans-serif !important;letter-spacing:-.03em;color:{text}}}
        .hero{{background:linear-gradient(135deg,rgba(20,33,61,.96),rgba(38,70,83,.92));color:#fff7ed;border-radius:28px;padding:2.2rem 2.4rem;margin-bottom:1rem}}
        .hero h1{{color:#fff8f2;font-size:3rem;margin:.5rem 0}}
        .hero p{{max-width:760px;line-height:1.65;color:rgba(255,248,242,.82)}}
        .hero-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem;margin-top:1rem}}
        .mini{{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:1rem}}
        .mini label{{display:block;font-size:.8rem;text-transform:uppercase;letter-spacing:.12em;color:rgba(255,248,242,.66)}}
        .mini strong{{font-family:'Space Grotesk',sans-serif;font-size:1.5rem}}
        .card{{background:{card};border:1px solid {card_border};border-radius:24px;padding:1.2rem 1.3rem;margin-bottom:1rem}}
        .eyebrow{{color:#ef8354;font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700}}
        .empty{{background:rgba(255,255,255,.84);border:1px dashed rgba(20,33,61,.18);border-radius:24px;padding:2rem;text-align:center}}
        .pill{{display:inline-block;background:rgba(42,157,143,.12);color:#2a9d8f;border:1px solid rgba(42,157,143,.18);border-radius:999px;padding:.45rem .8rem;margin-right:.5rem;font-size:.86rem;font-weight:600}}
        div[data-testid="stMetric"]{{background:{metric_bg};border:1px solid {card_border};padding:1rem;border-radius:22px}}
        .stTabs [data-baseweb="tab"]{{background:rgba(255,255,255,.55);border:1px solid rgba(20,33,61,.1);border-radius:999px}}
        .stTabs [aria-selected="true"]{{background:#ef8354;color:white}}
        .recommend-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem;margin:1rem 0}}
        .recommend-card{{background:{card};border:1px solid {card_border};border-radius:20px;padding:1rem}}
        .recommend-card strong{{display:block;font-size:1rem;margin-top:.35rem}}
        .landing-grid{{display:grid;grid-template-columns:1.2fr .8fr;gap:1rem;margin:1rem 0}}
        .method-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_csv(uploaded, key, label):
    if uploaded is None:
        return st.session_state.get(key)
    try:
        df = pd.read_csv(uploaded)
        st.session_state[key] = df
        return df
    except Exception as exc:
        st.sidebar.error(f"{label} could not be read: {exc}")
        return st.session_state.get(key)


def valid(df, label, show=True):
    ok = df is not None and REQUIRED_TARGET in df.columns and len(df) > 0
    if show and not ok:
        st.error(f"{label} must include `{REQUIRED_TARGET}` and contain at least one row.")
    return ok


def build_dataset_summary(df, label):
    if df is None:
        return {"Dataset": label, "Rows": 0, "Columns": 0, "Fraud Rate (%)": 0.0, "Missing Cells": 0}
    return {
        "Dataset": label,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Fraud Rate (%)": round(df[REQUIRED_TARGET].mean() * 100, 2) if REQUIRED_TARGET in df.columns else 0.0,
        "Missing Cells": int(df.isna().sum().sum()),
    }


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=120, max_depth=5, learning_rate=0.08)
    return models


def energy(runtime, watts, util, intensity):
    kwh = (runtime * watts * util) / 3_600_000
    return round(kwh * 1000, 4), round(kwh * intensity, 4)


@st.cache_data(show_spinner=False)
def benchmark(training_sets, df_test, limit, watts, util, intensity, use_codecarbon):
    rows = []
    for dataset_name, train_df in training_sets.items():
        X_train, y_train, X_test, y_test = prepare_dataset_pair(train_df, df_test)
        X_train, y_train = sample_dataset(X_train, y_train, limit)
        X_test, y_test = sample_dataset(X_test, y_test, limit)
        if len(X_train) == 0 or len(X_test) == 0 or pd.Series(y_train).nunique() < 2:
            continue
        for model_name, template in build_models().items():
            model = clone(template)
            tracker = None
            if use_codecarbon and EmissionsTracker is not None:
                try:
                    tracker = EmissionsTracker(log_level="error", save_to_file=False)
                except Exception:
                    tracker = None
            start = time.perf_counter()
            if tracker is not None:
                try:
                    tracker.start()
                except Exception:
                    tracker = None
            model.fit(X_train, y_train)
            measured = None
            if tracker is not None:
                try:
                    emissions_kg = tracker.stop()
                    measured = (emissions_kg or 0.0) * 1000
                except Exception:
                    measured = None
            train_s = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(X_test)
            pred_s = time.perf_counter() - start
            report = classification_report(y_test, pred, output_dict=True, zero_division=0).get("1", {})
            train_wh, train_g = energy(train_s, watts, util, intensity)
            pred_wh, pred_g = energy(pred_s, watts, util, intensity)
            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Precision": round(report.get("precision", 0), 3),
                "Recall": round(report.get("recall", 0), 3),
                "F1 Score": round(report.get("f1-score", 0), 3),
                "Train Time (s)": round(train_s, 4),
                "Predict Time (s)": round(pred_s, 4),
                "Total Energy (Wh)": round(train_wh + pred_wh, 4),
                "Total Carbon (gCO2e)": round(train_g + pred_g, 4),
                "Measured Carbon (gCO2e)": round(measured, 4) if measured is not None else None,
            })
    return pd.DataFrame(rows).sort_values(["Dataset", "F1 Score"], ascending=[True, False]).reset_index(drop=True)


def section(title, copy, eyebrow="Section"):
    st.markdown(f'<div class="card"><div class="eyebrow">{eyebrow}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)


def hero(df_train, df_test):
    train_rows = len(df_train) if df_train is not None else 0
    test_rows = len(df_test) if df_test is not None else 0
    fraud_rate = df_train[REQUIRED_TARGET].mean() * 100 if valid(df_train, "", False) else 0
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow" style="color:#f8c9b4">AI Risk Intelligence Platform</div>
            <h1>Sentinel AI Fraud Command Center</h1>
            <p>A production-style workspace for fraud benchmarking, investigation routing, carbon-aware modeling, and stakeholder-ready analytics.</p>
            <div><span class="pill">Fraud benchmarking</span><span class="pill">Two-stage detection</span><span class="pill">Carbon-aware modeling</span></div>
            <div class="hero-grid">
                <div class="mini"><label>Training Rows</label><strong>{train_rows:,}</strong></div>
                <div class="mini"><label>Test Rows</label><strong>{test_rows:,}</strong></div>
                <div class="mini"><label>Observed Fraud Rate</label><strong>{fraud_rate:.2f}%</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def charts(results_df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    display = results_df.copy()
    display["Label"] = display["Dataset"] + " | " + display["Model"]
    x = range(len(display))
    ax.bar(x, display["Precision"], width=0.25, label="Precision", color="#2a9d8f")
    ax.bar([i + 0.25 for i in x], display["Recall"], width=0.25, label="Recall", color="#f4a261")
    ax.bar([i + 0.5 for i in x], display["F1 Score"], width=0.25, label="F1 Score", color="#e76f51")
    ax.set_xticks([i + 0.25 for i in x])
    ax.set_xticklabels(display["Label"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.bar_chart(results_df.pivot(index="Model", columns="Dataset", values="Total Carbon (gCO2e)").fillna(0), use_container_width=True)
    with right:
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        for _, row in results_df.iterrows():
            ax2.scatter(row["Total Energy (Wh)"], row["F1 Score"], s=90, color="#264653")
            ax2.text(row["Total Energy (Wh)"], row["F1 Score"], f'{row["Dataset"]} | {row["Model"]}', fontsize=8)
        ax2.set_xlabel("Estimated Total Energy (Wh)")
        ax2.set_ylabel("F1 Score")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)


def render_recommendations(results_df):
    best = results_df.sort_values("F1 Score", ascending=False).iloc[0]
    fastest = results_df.sort_values("Train Time (s)").iloc[0]
    greenest = results_df.sort_values("Total Carbon (gCO2e)").iloc[0]
    st.markdown(
        f"""
        <div class="recommend-grid">
            <div class="recommend-card">
                <div class="eyebrow">Best For Accuracy</div>
                <strong>{best["Model"]}</strong>
                <div>{best["Dataset"]} | F1 {best["F1 Score"]:.3f}</div>
            </div>
            <div class="recommend-card">
                <div class="eyebrow">Best For Speed</div>
                <strong>{fastest["Model"]}</strong>
                <div>{fastest["Dataset"]} | {fastest["Train Time (s)"]:.3f}s train</div>
            </div>
            <div class="recommend-card">
                <div class="eyebrow">Best For Carbon</div>
                <strong>{greenest["Model"]}</strong>
                <div>{greenest["Dataset"]} | {greenest["Total Carbon (gCO2e)"]:.3f} gCO2e</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_algorithm_emissions(results_df):
    section("Carbon footprint for each algorithm", "This view makes the emissions for every algorithm explicit so you can present them clearly.", "Algorithm Carbon")
    dataset_options = ["All Datasets"] + sorted(results_df["Dataset"].unique().tolist())
    selected_dataset = st.selectbox("Dataset scope for algorithm emissions", dataset_options, key="algorithm_emissions_scope")

    scoped_df = results_df if selected_dataset == "All Datasets" else results_df[results_df["Dataset"] == selected_dataset]
    emission_df = (
        scoped_df.groupby("Model", as_index=False)[
            ["Train Time (s)", "Predict Time (s)", "Total Energy (Wh)", "Total Carbon (gCO2e)", "Measured Carbon (gCO2e)"]
        ]
        .mean(numeric_only=True)
        .sort_values("Total Carbon (gCO2e)", ascending=False)
    )

    if emission_df.empty:
        st.info("No algorithm emission data is available for the selected scope.")
        return

    cards = st.columns(min(4, len(emission_df)))
    for idx, (_, row) in enumerate(emission_df.iterrows()):
        with cards[idx % len(cards)]:
            st.metric(
                row["Model"],
                f'{row["Total Carbon (gCO2e)"]:.3f} gCO2e',
                f'{row["Total Energy (Wh)"]:.3f} Wh',
            )
            if pd.notna(row["Measured Carbon (gCO2e)"]):
                st.caption(f'Measured: {row["Measured Carbon (gCO2e)"]:.3f} gCO2e')
            else:
                st.caption("Measured: estimate only")

    st.bar_chart(emission_df.set_index("Model")[["Total Carbon (gCO2e)"]], use_container_width=True)
    st.dataframe(emission_df, use_container_width=True, hide_index=True)


def render_alerts(df_train, df_test, results_df):
    alerts = []
    test_fraud_rate = float(df_test[REQUIRED_TARGET].mean() * 100)
    if test_fraud_rate > 5:
        alerts.append(f"Fraud rate in the test set is elevated at {test_fraud_rate:.2f}%.")
    low_recall = results_df["Recall"].max()
    if low_recall < 0.70:
        alerts.append(f"No algorithm crossed 0.70 recall. Current best recall is {low_recall:.3f}.")
    high_carbon = results_df["Total Carbon (gCO2e)"].max()
    if high_carbon > results_df["Total Carbon (gCO2e)"].median() * 1.5:
        alerts.append(f"Some runs are materially more carbon-intensive, peaking at {high_carbon:.3f} gCO2e.")
    train_fraud_rate = float(df_train[REQUIRED_TARGET].mean() * 100)
    if abs(train_fraud_rate - test_fraud_rate) > 2:
        alerts.append(f"Train/test fraud-rate shift detected: train {train_fraud_rate:.2f}% vs test {test_fraud_rate:.2f}%.")

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No immediate risk alerts were triggered by the current run.")


def render_curves(training_sets, df_test, dataset_name):
    section("ROC and precision-recall curves", "These evaluation curves show how well the algorithms separate fraud from legitimate traffic beyond a single threshold.", "Advanced Evaluation")
    X_train, y_train, X_test, y_test = prepare_dataset_pair(training_sets[dataset_name], df_test)
    X_train, y_train = sample_dataset(X_train, y_train, min(len(X_train), 12000))
    X_test, y_test = sample_dataset(X_test, y_test, min(len(X_test), 12000))

    if pd.Series(y_train).nunique() < 2:
        st.info("ROC and precision-recall curves need at least two classes in the training data.")
        return
    if len(np.unique(y_test)) < 2:
        st.info("ROC and precision-recall curves need both fraud and legitimate samples in the evaluation set.")
        return

    roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
    pr_fig, pr_ax = plt.subplots(figsize=(7, 5))

    for model_name, template in build_models().items():
        model = clone(template)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)

        roc_ax.plot(fpr, tpr, label=f"{model_name} (AUC {roc_auc:.3f})")
        pr_ax.plot(recall, precision, label=f"{model_name} (AUC {pr_auc:.3f})")

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend(fontsize=8)
    roc_fig.tight_layout()

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-Recall Curve")
    pr_ax.legend(fontsize=8)
    pr_fig.tight_layout()

    left, right = st.columns(2)
    with left:
        st.pyplot(roc_fig, use_container_width=True)
    with right:
        st.pyplot(pr_fig, use_container_width=True)


def render_feature_importance(training_sets, df_test, dataset_name):
    section("Feature importance", "See which inputs matter most for tree-based algorithms on the selected dataset.", "Explainability")
    X_train, y_train, X_test, _ = prepare_dataset_pair(training_sets[dataset_name], df_test)
    X_train, y_train = sample_dataset(X_train, y_train, min(len(X_train), 12000))
    if pd.Series(y_train).nunique() < 2:
        st.info("Feature importance needs at least two classes in the training data.")
        return
    options = [name for name in build_models().keys() if name in {"Random Forest", "XGBoost"}]
    if not options:
        st.info("No tree-based model with feature importance is available in this environment.")
        return
    model_name = st.selectbox("Feature importance model", options, key="feature_importance_model")
    model = clone(build_models()[model_name])
    model.fit(X_train, y_train)
    if not hasattr(model, "feature_importances_"):
        st.info("This selected model does not expose feature importances.")
        return
    importance = (
        pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(12)
    )
    st.dataframe(importance, use_container_width=True, hide_index=True)
    st.bar_chart(importance.set_index("Feature"), use_container_width=True)


def answer_query(query, df_test, df_bal, results_df):
    q = query.lower()
    if "best model" in q or "highest f1" in q:
        row = results_df.sort_values("F1 Score", ascending=False).iloc[0]
        return f'Best model is {row["Model"]} on {row["Dataset"]} with F1 {row["F1 Score"]}.'
    if "lowest carbon" in q or "greenest" in q:
        row = results_df.sort_values("Total Carbon (gCO2e)").iloc[0]
        return f'Lowest estimated carbon comes from {row["Model"]} on {row["Dataset"]} at {row["Total Carbon (gCO2e)"]} gCO2e.'
    if "fastest" in q:
        row = results_df.sort_values("Train Time (s)").iloc[0]
        return f'Fastest training run is {row["Model"]} on {row["Dataset"]} at {row["Train Time (s)"]} seconds.'
    if "fraud percentage" in q:
        return f'Fraud percentage in the test set: {(df_test[REQUIRED_TARGET].mean() * 100):.2f}%'
    if "legitimate" in q:
        legit = len(df_test) - int(df_test[REQUIRED_TARGET].sum())
        return f"Legitimate transactions in the test set: {legit}"
    if ("fraud type" in q or "common fraud" in q) and df_bal is not None and "category" in df_bal.columns:
        fraud_categories = df_bal[df_bal[REQUIRED_TARGET] == 1]["category"].value_counts()
        if not fraud_categories.empty:
            return f"Most common fraud type in the balanced dataset: {fraud_categories.idxmax()}"
        return "The balanced dataset does not currently contain any fraud rows with category values."
    return "Try asking about the best model, lowest carbon run, fastest run, fraud percentage, or common fraud type."


def render_landing_summary():
    st.markdown(
        """
        <div class="landing-grid">
            <div class="card">
                <div class="eyebrow">Launch Summary</div>
                <h3>Built to feel like a final product</h3>
                <p>This dashboard blends fraud analytics, carbon awareness, manual prediction, explainability, and stakeholder-facing summaries into one experience.</p>
            </div>
            <div class="card">
                <div class="eyebrow">What You Can Do</div>
                <p>Benchmark algorithms, inspect feature importance, compare carbon footprint, try manual predictions, and export results for reporting.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_methodology():
    section("How the system works", "A concise methodology view helps users trust the output and understand how carbon and model quality are being estimated.", "Methodology")
    st.markdown(
        """
        <div class="method-grid">
            <div class="card">
                <div class="eyebrow">Fraud Modeling</div>
                <p>Numeric features are extracted from the uploaded datasets, time fields are expanded into behavioral signals, and multiple models are benchmarked on aligned train/test feature sets.</p>
            </div>
            <div class="card">
                <div class="eyebrow">Two-Stage Flow</div>
                <p>Stage 1 rapidly filters likely legitimate traffic. Stage 2 applies a stronger model to suspicious transactions to improve fraud capture where it matters most.</p>
            </div>
            <div class="card">
                <div class="eyebrow">Carbon Estimation</div>
                <p>Estimated carbon uses runtime, assumed device power, utilization, and grid intensity. If CodeCarbon is installed, measured emissions are also shown when available.</p>
            </div>
            <div class="card">
                <div class="eyebrow">Interpretation</div>
                <p>Accuracy metrics, ROC/PR curves, feature importance, and alerts should be read together before choosing a production model.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.session_state["theme_mode"] = st.radio("Theme", ["Light", "Dark"], horizontal=True, index=0 if st.session_state.get("theme_mode", "Light") == "Light" else 1)

styles()

with st.sidebar:
    st.markdown("## Control Room")
    st.caption("Upload datasets and tune benchmark assumptions.")
    train_file = st.file_uploader("Training dataset", type=["csv"])
    test_file = st.file_uploader("Test dataset", type=["csv"])
    balanced_file = st.file_uploader("Balanced reference dataset", type=["csv"])
    st.markdown("---")
    watts = st.slider("Estimated device power (W)", 20, 400, 95, 5)
    util = st.slider("Average utilization", 0.1, 1.0, 0.65, 0.05)
    intensity = st.slider("Grid carbon intensity (gCO2e/kWh)", 50, 1000, 475, 25)
    limit = st.slider("Benchmark sample size", 1000, 30000, 12000, 1000)
    use_codecarbon = st.checkbox("Use CodeCarbon if available", value=EmissionsTracker is not None)

df_train = load_csv(train_file, "df_train", "Training dataset")
df_test = load_csv(test_file, "df_test", "Test dataset")
df_bal = load_csv(balanced_file, "df_bal", "Balanced dataset")

hero(df_train, df_test)
render_landing_summary()

if not valid(df_train, "Training dataset") or not valid(df_test, "Test dataset"):
    st.markdown('<div class="empty"><div class="eyebrow">Ready When You Are</div><h3>Upload the training and test CSVs in the sidebar</h3><p>The dashboard will benchmark models, compare carbon impact, inspect predictions, and show trend intelligence.</p></div>', unsafe_allow_html=True)
    st.stop()

training_sets = {"Original Train": df_train}
if valid(df_bal, "Balanced dataset", False):
    training_sets["Balanced Train"] = df_bal

results_df = benchmark(training_sets, df_test, limit, watts, util, intensity, use_codecarbon)
if results_df.empty:
    st.error("No benchmark results could be generated from the uploaded datasets.")
    st.stop()

global_best = results_df.sort_values("F1 Score", ascending=False).iloc[0]
global_low = results_df.sort_values("Total Carbon (gCO2e)").iloc[0]

tabs = st.tabs(["Command Center", "Model Lab", "Predictions", "Insights", "Sustainability", "Data Quality", "Trends", "Methodology"])

with tabs[0]:
    section("Operational readiness and dataset posture", "A quick read on transaction volume, fraud balance, and whether a balanced dataset changes the training picture.", "Overview")
    train_total, train_fraud, train_legit = len(df_train), int(df_train[REQUIRED_TARGET].sum()), len(df_train) - int(df_train[REQUIRED_TARGET].sum())
    test_total, test_fraud = len(df_test), int(df_test[REQUIRED_TARGET].sum())
    m = st.columns(4)
    m[0].metric("Train Rows", f"{train_total:,}")
    m[1].metric("Train Fraud", f"{train_fraud:,}", f"{(train_fraud / train_total) * 100:.2f}%")
    m[2].metric("Test Rows", f"{test_total:,}")
    m[3].metric("Legitimate Train", f"{train_legit:,}")
    compare = [{"Dataset": "Original Train", "Rows": train_total, "Fraud %": round((train_fraud / train_total) * 100, 2)}, {"Dataset": "Test", "Rows": test_total, "Fraud %": round((test_fraud / test_total) * 100, 2)}]
    if valid(df_bal, "Balanced dataset", False):
        compare.append({"Dataset": "Balanced Train", "Rows": len(df_bal), "Fraud %": round(df_bal[REQUIRED_TARGET].mean() * 100, 2)})
    c1, c2 = st.columns([1.1, 0.9])
    c1.dataframe(pd.DataFrame(compare), use_container_width=True, hide_index=True)
    c2.bar_chart(pd.DataFrame(compare).set_index("Dataset")[["Fraud %"]], use_container_width=True)
    section("Alerts", "These warnings surface unusual fraud behavior, performance gaps, or carbon-heavy runs.", "Monitoring")
    render_alerts(df_train, df_test, results_df)
    section("Investigation routing", "Stage 1 filters low-risk traffic, while Stage 2 focuses heavier modeling power on suspicious transactions.", "Two-Stage Flow")
    selected_dataset = st.selectbox("Training dataset for the two-stage pipeline", list(training_sets.keys()))
    X_train, y_train, X_test, y_test = prepare_dataset_pair(training_sets[selected_dataset], df_test)
    if pd.Series(y_train).nunique() < 2:
        st.info("The two-stage pipeline needs at least two classes in the selected training dataset.")
    else:
        stage1, stage2 = train_two_stage_model(X_train, y_train)
        final_pred, suspicious_mask = predict_two_stage(stage1, stage2, X_test)
        m2 = st.columns(4)
        m2[0].metric("Filtered Legitimate", int((~suspicious_mask).sum()))
        m2[1].metric("Sent to Stage 2", int(suspicious_mask.sum()))
        m2[2].metric("Stage 2 Model", stage2.__class__.__name__)
        m2[3].metric("Final Fraud Recall", f'{classification_report(y_test, final_pred, output_dict=True, zero_division=0).get("1", {}).get("recall", 0):.3f}')
        st.pyplot(plot_confusion_matrix(get_confusion_matrix(y_test, final_pred)), use_container_width=True)

with tabs[1]:
    section("Accuracy, runtime, and carbon comparison", "Choose the best launch candidate by weighing fraud performance against cost, speed, and environmental impact.", "Model Lab")
    render_recommendations(results_df)
    s = st.columns(4)
    best = results_df.sort_values("F1 Score", ascending=False).iloc[0]
    low = results_df.sort_values("Total Carbon (gCO2e)").iloc[0]
    fast = results_df.sort_values("Train Time (s)").iloc[0]
    eff = results_df.assign(f1_per_wh=results_df["F1 Score"] / results_df["Total Energy (Wh)"].clip(lower=0.0001)).sort_values("f1_per_wh", ascending=False).iloc[0]
    s[0].metric("Best F1", f'{best["F1 Score"]:.3f}', f'{best["Dataset"]} | {best["Model"]}')
    s[1].metric("Lowest Carbon", f'{low["Total Carbon (gCO2e)"]:.3f} g', f'{low["Dataset"]} | {low["Model"]}')
    s[2].metric("Fastest Train", f'{fast["Train Time (s)"]:.3f} s', f'{fast["Dataset"]} | {fast["Model"]}')
    s[3].metric("Best F1 per Wh", f'{eff["f1_per_wh"]:.2f}', f'{eff["Dataset"]} | {eff["Model"]}')
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.download_button("Download benchmark CSV", data=results_df.to_csv(index=False), file_name="fraud_benchmark_results.csv", mime="text/csv")
    charts(results_df)
    st.divider()
    render_algorithm_emissions(results_df)
    curve_dataset = st.selectbox("Dataset for ROC/PR curves", list(training_sets.keys()), key="curve_dataset")
    render_curves(training_sets, df_test, curve_dataset)
    render_feature_importance(training_sets, df_test, curve_dataset)

with tabs[2]:
    section("Inspect a single transaction", "Use benchmarked models to inspect one test record and explain the predicted fraud risk.", "Prediction Workbench")
    best = results_df.sort_values("F1 Score", ascending=False).iloc[0]
    model_names = list(build_models().keys())
    dataset_names = list(training_sets.keys())
    a, b, c = st.columns(3)
    dataset_name = a.selectbox("Training dataset", dataset_names, index=dataset_names.index(best["Dataset"]))
    model_name = b.selectbox("Prediction model", model_names, index=model_names.index(best["Model"]))
    row_id = c.number_input("Test row index", min_value=0, max_value=max(len(df_test) - 1, 0), value=0, step=1)
    X_train, y_train, X_test, y_test = prepare_dataset_pair(training_sets[dataset_name], df_test)
    if pd.Series(y_train).nunique() < 2:
        st.info("Prediction workbench needs at least two classes in the selected training dataset.")
    else:
        model = clone(build_models()[model_name])
        model.fit(X_train, y_train)
        row = X_test.iloc[[int(row_id)]]
        pred = int(model.predict(row)[0])
        l, r = st.columns([0.8, 1.2])
        l.metric("Predicted Class", "Fraud" if pred == 1 else "Legitimate")
        l.metric("Actual Class", "Fraud" if int(y_test.iloc[int(row_id)]) == 1 else "Legitimate")
        if hasattr(model, "predict_proba"):
            l.metric("Fraud Probability", f'{float(model.predict_proba(row)[0][1]):.2%}')
        preview = row.T.reset_index()
        preview.columns = ["Feature", "Value"]
        r.dataframe(preview.head(20), use_container_width=True, hide_index=True)

    st.divider()
    section("Try a prediction", "Enter a simple transaction profile and run it through the Stage 2 fraud model.", "Manual Prediction")
    form_cols_top = st.columns(4)
    amt = form_cols_top[0].number_input("Transaction Amount", value=100.0, key="manual_amt")
    hour = form_cols_top[1].slider("Transaction Hour", 0, 23, 12, key="manual_hour")
    day = form_cols_top[2].slider("Day", 1, 31, 15, key="manual_day")
    month = form_cols_top[3].slider("Month", 1, 12, 6, key="manual_month")
    form_cols_bottom = st.columns(4)
    day_of_week = form_cols_bottom[0].slider("Day Of Week", 0, 6, 2, key="manual_dow")
    is_weekend = form_cols_bottom[1].selectbox("Weekend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="manual_weekend")
    merchant_risk = form_cols_bottom[2].slider("Merchant Risk Score", 0.0, 1.0, 0.25, 0.01, key="manual_merchant_risk")
    category_risk = form_cols_bottom[3].slider("Category Risk Score", 0.0, 1.0, 0.25, 0.01, key="manual_category_risk")

    if st.button("Predict Fraud", key="manual_predict"):
        manual_X_train, manual_y_train, _, _ = prepare_dataset_pair(training_sets[dataset_name], df_test)
        if pd.Series(manual_y_train).nunique() < 2:
            st.info("Manual prediction needs at least two classes in the selected training dataset.")
        else:
            _, manual_stage2 = train_two_stage_model(manual_X_train, manual_y_train)

            input_df = pd.DataFrame(
                [
                    {
                        "amt": amt,
                        "hour": hour,
                        "day": day,
                        "month": month,
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "merchant_risk": merchant_risk,
                        "category_risk": category_risk,
                    }
                ]
            )
            for col in manual_X_train.columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[manual_X_train.columns]
            pred = manual_stage2.predict(input_df)
            probability = None
            if hasattr(manual_stage2, "predict_proba"):
                probability = float(manual_stage2.predict_proba(input_df)[0][1])

            if int(pred[0]) == 1:
                st.error("Fraudulent Transaction Detected")
            else:
                st.success("Legitimate Transaction")
            if probability is not None:
                st.metric("Fraud Probability", f"{probability:.2%}")
                st.progress(int(np.clip(probability, 0, 1) * 100) / 100)

with tabs[3]:
    section("Ask the dashboard", "Give users a simple business-friendly layer on top of the raw metrics so they can pull answers quickly.", "Insights")
    query = st.text_input("Ask a question", placeholder="Best model, lowest carbon run, fraud percentage...")
    if query:
        st.success(answer_query(query, df_test, df_bal, results_df))

with tabs[4]:
    section("Runtime-aware carbon reporting", "Combine benchmark estimates with optional CodeCarbon measurements and historical emissions data when available.", "Sustainability")
    carbon = carbon_footprint()
    c = st.columns(4)
    c[0].metric("Energy Usage", carbon["energy"])
    c[1].metric("Carbon Impact", carbon["carbon"])
    c[2].metric("Greenest Run", global_low["Model"])
    c[3].metric("Best Model", global_best["Model"])
    try:
        score = sustainability_score(float(str(carbon["energy"]).split()[0]), float(str(carbon["carbon"]).split()[0]))
        st.progress(score / 100)
        st.success(f"Sustainability score: {score}/100")
    except Exception:
        st.info("Tracked emissions data was not available, so the sustainability score could not be computed yet.")
    carbon_df = get_carbon_data()
    if carbon_df is not None:
        cols = [col for col in ["energy_consumed", "emissions"] if col in carbon_df.columns]
        if cols:
            st.line_chart(carbon_df[cols], use_container_width=True)

with tabs[5]:
    section("Readiness checks before launch", "Inspect dataset health, missing values, and sample rows so the dashboard is grounded in trustworthy inputs.", "Data Quality")
    quality = pd.DataFrame([
        {"Dataset": "Training", "Rows": len(df_train), "Columns": len(df_train.columns), "Fraud Rate (%)": round(df_train[REQUIRED_TARGET].mean() * 100, 2), "Missing Cells": int(df_train.isna().sum().sum())},
        {"Dataset": "Test", "Rows": len(df_test), "Columns": len(df_test.columns), "Fraud Rate (%)": round(df_test[REQUIRED_TARGET].mean() * 100, 2), "Missing Cells": int(df_test.isna().sum().sum())},
        build_dataset_summary(df_bal, "Balanced Reference"),
    ])
    st.dataframe(quality, use_container_width=True, hide_index=True)
    preview_choice = st.selectbox("Preview dataset", ["Training", "Test", "Balanced Reference"])
    preview_map = {"Training": df_train, "Test": df_test, "Balanced Reference": df_bal}
    if preview_map[preview_choice] is not None:
        st.dataframe(preview_map[preview_choice].head(12), use_container_width=True)

with tabs[6]:
    section("Fraud movement and scaling pressure", "Track how fraud evolves over time and how larger training datasets can increase the computation footprint.", "Trend Intelligence")
    left, right = st.columns(2)
    with left:
        if "trans_date_trans_time" not in df_train.columns:
            st.info("Training data needs `trans_date_trans_time` for trend analysis.")
        else:
            trend_df = add_time_features(df_train).dropna(subset=["trans_date_trans_time"])
            trend_df["quarter"] = trend_df["trans_date_trans_time"].dt.to_period("Q").astype(str)
            left.line_chart(trend_df.groupby("quarter")[REQUIRED_TARGET].mean(), use_container_width=True)
    with right:
        scaling = [{"Dataset": "Original Train", "Rows": len(df_train)}]
        if valid(df_bal, "Balanced dataset", False):
            scaling.append({"Dataset": "Balanced Train", "Rows": len(df_bal)})
        right.bar_chart(pd.DataFrame(scaling).set_index("Dataset"), use_container_width=True)
        st.caption("More rows usually mean more computation, which often increases runtime and carbon cost.")

with tabs[7]:
    render_methodology()
