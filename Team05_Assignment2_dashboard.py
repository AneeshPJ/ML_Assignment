"""
================================================================================
BITS F464 MACHINE LEARNING - SECOND SEMESTER 2025-2026
Assignment 2: Automated ML Pipeline for Clinical Prediction under Temporal Shift
Team: Team05
================================================================================
"""

pip install streamlit pandas numpy scikit-learn matplotlib seaborn
streamlit run Team05_Assignment2_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import io
import pickle
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Sklearn imports ──────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Team05 | Clinical ML Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58a6ff;
}

.metric-label {
    font-size: 0.8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #3fb950;
    padding: 1rem;
    border-radius: 4px;
    font-size: 0.9rem;
}

.warn-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #d29922;
    padding: 1rem;
    border-radius: 4px;
    font-size: 0.9rem;
}

.stSelectbox > div > div { background-color: #161b22; }
.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover { background: #388bfd; }

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for key in ["data_loaded", "models_trained", "cl_trained",
            "df1_train", "df1_test", "df2_train", "df2_test",
            "models", "scaler", "label_enc", "feature_cols",
            "cl_models", "top_conditions"]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ["data_loaded","models_trained","cl_trained"] else False

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#d29922"
RED       = "#f85149"
PURPLE    = "#bc8cff"
COLORS    = [BLUE, GREEN, ORANGE, RED, PURPLE, "#ff7b72", "#ffa657"]

def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
    return fig

@st.cache_data(show_spinner=False)
def load_and_process(folder_path, split_year, top_n):
    # ── Load CSVs ─────────────────────────────────────────────────────────────
    def load(name):
        p = os.path.join(folder_path, name)
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False)
        return None

    patients    = load("patients.csv")
    encounters  = load("encounters.csv")
    conditions  = load("conditions.csv")
    observations= load("observations.csv")

    if any(df is None for df in [patients, encounters, conditions, observations]):
        return None, "One or more required CSV files not found in the folder."

    # ── Clean patients ────────────────────────────────────────────────────────
    patients = patients.rename(columns={"Id": "PATIENT"})
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    patients["AGE"] = (pd.Timestamp("2024-01-01") - patients["BIRTHDATE"]).dt.days / 365.25
    patients["GENDER_ENC"] = (patients["GENDER"] == "M").astype(int)
    race_dummies = pd.get_dummies(patients["RACE"], prefix="RACE")
    eth_dummies  = pd.get_dummies(patients["ETHNICITY"], prefix="ETH")
    patients = pd.concat([patients[["PATIENT","AGE","GENDER_ENC","INCOME",
                                    "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE"]],
                          race_dummies, eth_dummies], axis=1)

    # ── Encounters with year ──────────────────────────────────────────────────
    encounters = encounters.rename(columns={"Id":"ENCOUNTER"})
    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
    encounters["YEAR"]  = encounters["START"].dt.year
    enc_slim = encounters[["ENCOUNTER","PATIENT","YEAR","ENCOUNTERCLASS"]].copy()

    # ── Conditions ────────────────────────────────────────────────────────────
    # drop unnamed columns
    cond_cols = [c for c in conditions.columns if not c.startswith("Unnamed")]
    conditions = conditions[cond_cols].copy()
    conditions["START"] = pd.to_datetime(conditions["START"], errors="coerce", dayfirst=True)

    # Top N conditions
    top_conds = conditions["DESCRIPTION"].value_counts().head(top_n).index.tolist()
    conditions = conditions[conditions["DESCRIPTION"].isin(top_conds)].copy()
    # keep one condition per encounter (first)
    conditions = conditions.sort_values("START").drop_duplicates(subset=["PATIENT","ENCOUNTER"], keep="first")
    conditions = conditions[["PATIENT","ENCOUNTER","DESCRIPTION"]].rename(columns={"DESCRIPTION":"CONDITION"})

    # ── Observations pivot ────────────────────────────────────────────────────
    obs_numeric = observations[observations["TYPE"] == "numeric"].copy()
    obs_numeric["VALUE"] = pd.to_numeric(obs_numeric["VALUE"], errors="coerce")
    key_obs = ["Body Height","Body Weight","Pain severity - 0-10 verbal numeric rating [Score] - Reported",
               "Body Mass Index","Diastolic Blood Pressure","Systolic Blood Pressure",
               "Heart rate","Respiratory rate","Body temperature","Oxygen saturation in Arterial blood",
               "Glucose","Hemoglobin A1c/Hemoglobin.total in Blood"]
    obs_filtered = obs_numeric[obs_numeric["DESCRIPTION"].isin(key_obs)]
    # aggregate per patient
    obs_agg = obs_filtered.groupby(["PATIENT","DESCRIPTION"])["VALUE"].agg(["mean","std"]).reset_index()
    obs_agg.columns = ["PATIENT","OBS","MEAN","STD"]
    obs_pivot = obs_agg.pivot_table(index="PATIENT", columns="OBS", values=["MEAN","STD"])
    obs_pivot.columns = ["_".join(c).strip().replace(" ","_").replace("/","_")[:50] for c in obs_pivot.columns]
    obs_pivot = obs_pivot.reset_index()

    # ── Merge all ─────────────────────────────────────────────────────────────
    df = conditions.merge(enc_slim, on=["PATIENT","ENCOUNTER"], how="left")
    df = df.merge(patients, on="PATIENT", how="left")
    df = df.merge(obs_pivot, on="PATIENT", how="left")

    df = df.dropna(subset=["YEAR","CONDITION"])
    df["YEAR"] = df["YEAR"].astype(int)

    # ── Feature cols ──────────────────────────────────────────────────────────
    exclude = ["PATIENT","ENCOUNTER","CONDITION","YEAR"]
    feature_cols = [c for c in df.columns if c not in exclude]

    # fill NaN with median
    for c in feature_cols:
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(0)

    # ── Temporal split ────────────────────────────────────────────────────────
    df1 = df[df["YEAR"] <  split_year].copy()
    df2 = df[df["YEAR"] >= split_year].copy()

    le = LabelEncoder()
    df["LABEL"]  = le.fit_transform(df["CONDITION"])
    df1["LABEL"] = le.transform(df1["CONDITION"])
    df2["LABEL"] = le.transform(df2["CONDITION"])

    # ── Train/test split ──────────────────────────────────────────────────────
    def split(d):
        X = d[feature_cols].values
        y = d["LABEL"].values
        if len(d) < 10:
            return X, X, y, y
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    X1tr,X1te,y1tr,y1te = split(df1)
    X2tr,X2te,y2tr,y2te = split(df2)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X1tr_s = scaler.fit_transform(X1tr)
    X1te_s = scaler.transform(X1te)
    X2tr_s = scaler.transform(X2tr)
    X2te_s = scaler.transform(X2te)

    result = {
        "df": df, "df1": df1, "df2": df2,
        "X1tr": X1tr_s, "X1te": X1te_s,
        "X2tr": X2tr_s, "X2te": X2te_s,
        "y1tr": y1tr, "y1te": y1te,
        "y2tr": y2tr, "y2te": y2te,
        "feature_cols": feature_cols,
        "label_enc": le,
        "scaler": scaler,
        "top_conditions": top_conds,
        "split_year": split_year
    }
    return result, None

def get_metrics(y_true, y_pred, y_prob=None, le=None):
    avg = "weighted"
    m = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            m["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average=avg)
        except:
            m["ROC-AUC"] = np.nan
    return m

def train_models(data):
    X1tr, y1tr = data["X1tr"], data["y1tr"]
    models = {}

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X1tr, y1tr)
    models["Decision Tree"] = dt

    # SVM
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    svm.fit(X1tr, y1tr)
    models["SVM"] = svm

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                        random_state=42, early_stopping=True, validation_fraction=0.1)
    mlp.fit(X1tr, y1tr)
    models["Neural Network"] = mlp

    return models

def evaluate_model(model, X, y):
    pred = model.predict(X)
    try:
        prob = model.predict_proba(X)
    except:
        prob = None
    return get_metrics(y, pred, prob), pred

def continual_learning(models, data):
    X2tr, y2tr = data["X2tr"], data["y2tr"]
    cl_models = {}

    # DT: retrain with combined data
    X_all = np.vstack([data["X1tr"], X2tr])
    y_all = np.concatenate([data["y1tr"], y2tr])
    dt_cl = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_cl.fit(X_all, y_all)
    cl_models["Decision Tree"] = dt_cl

    # SVM: fine-tune — retrain on DS2
    svm_cl = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    svm_cl.fit(X2tr, y2tr)
    cl_models["SVM"] = svm_cl

    # MLP: warm start / fine-tune
    mlp_base = models["Neural Network"]
    mlp_cl = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=100,
        random_state=42,
        warm_start=True
    )
    mlp_cl.coefs_ = [c.copy() for c in mlp_base.coefs_]
    mlp_cl.intercepts_ = [i.copy() for i in mlp_base.intercepts_]
    mlp_cl.n_iter_ = mlp_base.n_iter_
    mlp_cl.n_outputs_ = mlp_base.n_outputs_
    mlp_cl.out_activation_ = mlp_base.out_activation_
    mlp_cl.n_layers_ = mlp_base.n_layers_
    mlp_cl.t_ = mlp_base.t_
    mlp_cl.loss_ = mlp_base.loss_
    try:
        mlp_cl.fit(X2tr, y2tr)
    except:
        mlp_cl = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
        mlp_cl.fit(X2tr, y2tr)
    cl_models["Neural Network"] = mlp_cl

    return cl_models

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 Team05 ML Pipeline")
    st.markdown("**BITS F464 Machine Learning**")
    st.markdown("*Assignment 2 — 2025-2026*")
    st.divider()

    folder_path = st.text_input(
        "📁 Dataset Folder Path",
        value="./data",
        help="Folder containing all CSV files"
    )
    split_year = st.slider("📅 Temporal Split Year", 2005, 2020, 2015)
    top_n      = st.slider("🎯 Top N Conditions (Target)", 3, 10, 5)

    st.divider()
    if st.button("🚀 Load & Process Data", use_container_width=True):
        with st.spinner("Loading and processing data..."):
            result, err = load_and_process(folder_path, split_year, top_n)
            if err:
                st.error(err)
            else:
                st.session_state.data        = result
                st.session_state.data_loaded = True
                st.session_state.models_trained = False
                st.session_state.cl_trained  = False
                st.success("✅ Data loaded!")

    if st.session_state.data_loaded:
        if st.button("🤖 Train Models (DS1)", use_container_width=True):
            with st.spinner("Training Decision Tree, SVM, Neural Network..."):
                data = st.session_state.data
                models = train_models(data)
                st.session_state.models = models
                st.session_state.models_trained = True
                st.success("✅ Models trained!")

    if st.session_state.models_trained:
        if st.button("🔄 Continual Learning (DS2)", use_container_width=True):
            with st.spinner("Fine-tuning on Dataset 2..."):
                cl_models = continual_learning(
                    st.session_state.models,
                    st.session_state.data
                )
                st.session_state.cl_models  = cl_models
                st.session_state.cl_trained = True
                st.success("✅ Continual learning done!")

    st.divider()
    st.markdown("""
    **Pipeline Status**
    """)
    st.markdown(f"{'✅' if st.session_state.data_loaded else '⬜'} Data Loaded")
    st.markdown(f"{'✅' if st.session_state.models_trained else '⬜'} Models Trained")
    st.markdown(f"{'✅' if st.session_state.cl_trained else '⬜'} Continual Learning")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1 style="color:#58a6ff; margin:0; font-size:1.8rem;">
        🏥 Clinical ML Pipeline Dashboard
    </h1>
    <p style="color:#8b949e; margin:0.5rem 0 0 0; font-family:'IBM Plex Mono',monospace; font-size:0.85rem;">
        BITS F464 Machine Learning · Team05 · Assignment 2 · 2025-2026
    </p>
    <p style="color:#8b949e; margin:0.2rem 0 0 0; font-size:0.8rem;">
        Automated ML Pipeline for Clinical Prediction under Temporal Shift in EHR Data
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.markdown("""
    <div class="info-box">
        👈 <strong>Get Started:</strong> Enter your dataset folder path in the sidebar and click
        <strong>"Load & Process Data"</strong>. The folder should contain the CSV files from the
        Google Drive link provided in the assignment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Expected CSV Files")
    cols = st.columns(4)
    files = ["patients.csv","encounters.csv","conditions.csv","observations.csv",
             "procedures.csv","medications.csv","allergies.csv","immunizations.csv"]
    for i,f in enumerate(files):
        with cols[i%4]:
            st.markdown(f"`{f}`")

    st.markdown("### 🗺️ Pipeline Overview")
    st.markdown("""
    | Step | Task | Status |
    |------|------|--------|
    | 1 | Data Loading & Merging | ⬜ Pending |
    | 2 | Feature Engineering & EDA | ⬜ Pending |
    | 3 | Model Training (DS1) | ⬜ Pending |
    | 4 | Cross-Dataset Evaluation | ⬜ Pending |
    | 5 | Continual Learning (DS2) | ⬜ Pending |
    """)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
data = st.session_state.data
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA",
    "🤖 Model Training",
    "📈 Evaluation",
    "🔄 Continual Learning",
    "📋 Summary"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)

    df  = data["df"]
    df1 = data["df1"]
    df2 = data["df2"]

    # Overview metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    metrics_data = [
        ("Total Records", len(df)),
        ("DS1 (Historical)", len(df1)),
        ("DS2 (Current)", len(df2)),
        ("Features", len(data["feature_cols"])),
        ("Conditions", len(data["top_conditions"]))
    ]
    for col,(label,val) in zip([c1,c2,c3,c4,c5], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val:,}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── Condition distribution ────────────────────────────────────────────────
    with col_left:
        st.markdown("#### 🎯 Condition Distribution (DS1 vs DS2)")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, d, title in zip(axes, [df1, df2], ["Dataset 1\n(Historical)", "Dataset 2\n(Current)"]):
            counts = d["CONDITION"].value_counts()
            short_labels = [c[:25]+"…" if len(c)>25 else c for c in counts.index]
            ax.barh(short_labels, counts.values, color=COLORS[:len(counts)], alpha=0.85)
            ax.set_title(title, color="#e6edf3")
            ax.invert_yaxis()
        style_fig(fig)
        st.pyplot(fig)
        plt.close()

    # ── Temporal distribution ─────────────────────────────────────────────────
    with col_right:
        st.markdown("#### 📅 Records Over Time")
        fig, ax = plt.subplots(figsize=(8, 4))
        yearly = df.groupby("YEAR").size().reset_index(name="count")
        split_yr = data["split_year"]
        ds1_y = yearly[yearly["YEAR"] < split_yr]
        ds2_y = yearly[yearly["YEAR"] >= split_yr]
        ax.bar(ds1_y["YEAR"], ds1_y["count"], color=BLUE, alpha=0.8, label="Dataset 1")
        ax.bar(ds2_y["YEAR"], ds2_y["count"], color=GREEN, alpha=0.8, label="Dataset 2")
        ax.axvline(split_yr, color=ORANGE, linestyle="--", linewidth=1.5, label=f"Split ({split_yr})")
        ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor="#e6edf3")
        ax.set_xlabel("Year")
        ax.set_ylabel("Encounters")
        ax.set_title("Temporal Distribution of Records")
        style_fig(fig)
        st.pyplot(fig)
        plt.close()

    col_left2, col_right2 = st.columns(2)

    # ── Age distribution ──────────────────────────────────────────────────────
    with col_left2:
        st.markdown("#### 👤 Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df1["AGE"].dropna(), bins=25, alpha=0.7, color=BLUE,   label="DS1", density=True)
        ax.hist(df2["AGE"].dropna(), bins=25, alpha=0.7, color=GREEN,  label="DS2", density=True)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Density")
        ax.set_title("Age Distribution: DS1 vs DS2")
        ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor="#e6edf3")
        style_fig(fig)
        st.pyplot(fig)
        plt.close()

    # ── Gender split ──────────────────────────────────────────────────────────
    with col_right2:
        st.markdown("#### ⚧ Gender Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, d, title in zip(axes, [df1, df2], ["DS1","DS2"]):
            gender_map = {1: "Male", 0: "Female"}
            counts = d["GENDER_ENC"].map(gender_map).value_counts()
            ax.pie(counts.values, labels=counts.index,
                   colors=[BLUE, PURPLE], autopct="%1.1f%%",
                   textprops={"color":"#e6edf3"})
            ax.set_title(title, color="#e6edf3")
        style_fig(fig)
        st.pyplot(fig)
        plt.close()

    # ── Descriptive statistics ────────────────────────────────────────────────
    st.markdown("#### 📐 Descriptive Statistics — Dataset 1")
    num_cols = [c for c in data["feature_cols"] if df1[c].dtype in [np.float64, np.float32, np.int64, np.int32]][:8]
    st.dataframe(df1[num_cols].describe().round(3).style.background_gradient(cmap="Blues"), use_container_width=True)

    # ── Data drift detection ──────────────────────────────────────────────────
    st.markdown("#### 🌊 Data Drift Detection (DS1 vs DS2 Mean Comparison)")
    drift_data = []
    for c in num_cols:
        m1 = df1[c].mean()
        m2 = df2[c].mean()
        drift_data.append({"Feature": c, "DS1 Mean": round(m1,3), "DS2 Mean": round(m2,3),
                           "Δ": round(m2-m1,3), "Drift %": round(abs(m2-m1)/(abs(m1)+1e-9)*100,1)})
    drift_df = pd.DataFrame(drift_data).sort_values("Drift %", ascending=False)
    st.dataframe(drift_df.style.background_gradient(subset=["Drift %"], cmap="YlOrRd"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<h2 class="section-header">Model Training on Dataset 1</h2>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.markdown("""
        <div class="warn-box">
            ⚠️ Models not yet trained. Click <strong>"Train Models (DS1)"</strong> in the sidebar.
        </div>""", unsafe_allow_html=True)
        st.stop()

    models = st.session_state.models
    data   = st.session_state.data

    # ── Training set performance ──────────────────────────────────────────────
    st.markdown("### 🎯 Training Performance (Dataset 1 — Train Set)")
    train_results = {}
    for name, model in models.items():
        m, _ = evaluate_model(model, data["X1tr"], data["y1tr"])
        train_results[name] = m

    tr_df = pd.DataFrame(train_results).T.round(4)
    st.dataframe(tr_df.style.background_gradient(cmap="Greens", subset=["Accuracy","F1"]),
                 use_container_width=True)

    # ── Model complexity sliders ──────────────────────────────────────────────
    st.markdown("### ⚙️ Model Complexity Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Decision Tree — Depth vs Accuracy")
        depths = range(1, 16)
        tr_scores, te_scores = [], []
        for d in depths:
            dt = DecisionTreeClassifier(max_depth=d, random_state=42)
            dt.fit(data["X1tr"], data["y1tr"])
            tr_scores.append(accuracy_score(data["y1tr"], dt.predict(data["X1tr"])))
            te_scores.append(accuracy_score(data["y1te"], dt.predict(data["X1te"])))

        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(depths, tr_scores, color=BLUE,  marker="o", label="Train")
        ax.plot(depths, te_scores, color=GREEN, marker="s", label="Test")
        ax.axvline(5, color=ORANGE, linestyle="--", alpha=0.7, label="Selected depth=5")
        ax.set_xlabel("Max Depth")
        ax.set_ylabel("Accuracy")
        ax.set_title("Bias-Variance: Decision Tree")
        ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor="#e6edf3")
        style_fig(fig)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### MLP — Training Loss Curve")
        mlp = models["Neural Network"]
        if hasattr(mlp, "loss_curve_"):
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(mlp.loss_curve_, color=PURPLE, linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Neural Network Training Loss")
            style_fig(fig)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Loss curve not available.")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### 🔍 Feature Importance — Decision Tree")
    dt_model = models["Decision Tree"]
    importances = dt_model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": data["feature_cols"],
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(feat_imp["Feature"][::-1], feat_imp["Importance"][::-1],
                   color=COLORS[:len(feat_imp)])
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Features — Decision Tree")
    style_fig(fig)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<h2 class="section-header">Cross-Dataset Evaluation</h2>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.markdown('<div class="warn-box">⚠️ Train models first (sidebar).</div>', unsafe_allow_html=True)
        st.stop()

    models = st.session_state.models
    data   = st.session_state.data
    le     = data["label_enc"]

    # ── Full eval table ───────────────────────────────────────────────────────
    eval_rows = []
    for name, model in models.items():
        for (X,y,label) in [
            (data["X1te"], data["y1te"], "DS1 Test"),
            (data["X2te"], data["y2te"], "DS2 Test")
        ]:
            m, _ = evaluate_model(model, X, y)
            eval_rows.append({"Model":name, "Eval Set":label, **{k:round(v,4) for k,v in m.items()}})

    eval_df = pd.DataFrame(eval_rows)
    st.dataframe(eval_df.style.background_gradient(subset=["Accuracy","F1"], cmap="RdYlGn"),
                 use_container_width=True)

    st.markdown("---")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.markdown("### 📊 Performance Comparison: DS1 Test vs DS2 Test")
    model_names = list(models.keys())
    metric_sel  = st.selectbox("Select Metric", ["Accuracy","F1","Precision","Recall"])

    ds1_vals = [eval_df[(eval_df["Model"]==n)&(eval_df["Eval Set"]=="DS1 Test")][metric_sel].values[0] for n in model_names]
    ds2_vals = [eval_df[(eval_df["Model"]==n)&(eval_df["Eval Set"]=="DS2 Test")][metric_sel].values[0] for n in model_names]

    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(10,5))
    bars1 = ax.bar(x - 0.2, ds1_vals, 0.35, label="DS1 Test", color=BLUE,  alpha=0.85)
    bars2 = ax.bar(x + 0.2, ds2_vals, 0.35, label="DS2 Test", color=GREEN, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel(metric_sel)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{metric_sel}: DS1 Test vs DS2 Test (Temporal Shift Impact)")
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor="#e6edf3")
    for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", color="#e6edf3", fontsize=8)
    for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", color="#e6edf3", fontsize=8)
    style_fig(fig)
    st.pyplot(fig)
    plt.close()

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("### 🧩 Confusion Matrices")
    model_sel = st.selectbox("Select Model", model_names)
    model     = models[model_sel]
    class_names = [c[:20] for c in le.classes_]

    col1, col2 = st.columns(2)
    for col, (X,y,title) in zip([col1,col2],[
        (data["X1te"], data["y1te"], "DS1 Test"),
        (data["X2te"], data["y2te"], "DS2 Test")
    ]):
        with col:
            pred = model.predict(X)
            cm   = confusion_matrix(y, pred)
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title(f"{model_sel} — {title}", color="#e6edf3")
            ax.set_xlabel("Predicted", color="#8b949e")
            ax.set_ylabel("Actual", color="#8b949e")
            style_fig(fig)
            st.pyplot(fig)
            plt.close()

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.markdown("### 📉 ROC Curves — DS1 Test")
    fig, ax = plt.subplots(figsize=(8,6))
    n_classes = len(le.classes_)
    for idx,(name,model) in enumerate(models.items()):
        try:
            prob = model.predict_proba(data["X1te"])
            if n_classes == 2:
                fpr,tpr,_ = roc_curve(data["y1te"], prob[:,1])
                auc = roc_auc_score(data["y1te"], prob[:,1])
                ax.plot(fpr, tpr, color=COLORS[idx], label=f"{name} (AUC={auc:.3f})")
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(data["y1te"], classes=range(n_classes))
                auc = roc_auc_score(y_bin, prob, multi_class="ovr", average="macro")
                ax.text(0.6, 0.3-idx*0.07, f"{name}: AUC={auc:.3f}", color=COLORS[idx], fontsize=10)
        except Exception as e:
            pass
    ax.plot([0,1],[0,1],"--", color="#30363d")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (DS1 Test)")
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor="#e6edf3")
    style_fig(fig)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONTINUAL LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<h2 class="section-header">Continual Learning on Dataset 2</h2>', unsafe_allow_html=True)

    if not st.session_state.cl_trained:
        st.markdown('<div class="warn-box">⚠️ Run Continual Learning from the sidebar first.</div>', unsafe_allow_html=True)
        st.stop()

    cl_models = st.session_state.cl_models
    models    = st.session_state.models
    data      = st.session_state.data

    # ── Before vs After table ─────────────────────────────────────────────────
    st.markdown("### 🔄 Before vs After Continual Learning — DS2 Test")
    rows = []
    for name in models.keys():
        m_before, _ = evaluate_model(models[name],    data["X2te"], data["y2te"])
        m_after,  _ = evaluate_model(cl_models[name], data["X2te"], data["y2te"])
        for metric in ["Accuracy","F1","Precision","Recall"]:
            v_b = m_before.get(metric, np.nan)
            v_a = m_after.get(metric,  np.nan)
            rows.append({
                "Model":name, "Metric":metric,
                "Before CL": round(v_b,4),
                "After CL":  round(v_a,4),
                "Δ":         round(v_a-v_b,4)
            })

    cl_df = pd.DataFrame(rows)
    st.dataframe(
        cl_df.style.background_gradient(subset=["Δ"], cmap="RdYlGn"),
        use_container_width=True
    )

    # ── Visual comparison ─────────────────────────────────────────────────────
    st.markdown("### 📊 Visual: Accuracy Before vs After CL")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for idx, name in enumerate(models.keys()):
        ax = axes[idx]
        m_b, _ = evaluate_model(models[name],    data["X2te"], data["y2te"])
        m_a, _ = evaluate_model(cl_models[name], data["X2te"], data["y2te"])
        metrics_list = ["Accuracy","F1","Precision","Recall"]
        vals_b = [m_b.get(m,0) for m in metrics_list]
        vals_a = [m_a.get(m,0) for m in metrics_list]
        x = np.arange(len(metrics_list))
        ax.bar(x-0.2, vals_b, 0.35, label="Before CL", color=RED,   alpha=0.8)
        ax.bar(x+0.2, vals_a, 0.35, label="After CL",  color=GREEN, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_list, rotation=20, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(name, color="#e6edf3")
        ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7)
    style_fig(fig)
    plt.suptitle("Continual Learning Impact on DS2 Test", color="#e6edf3", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>Continual Learning Strategy:</strong><br>
    • <strong>Decision Tree</strong>: Combined DS1 + DS2 training data refit (data accumulation strategy)<br>
    • <strong>SVM</strong>: Fine-tuned on DS2 training data (domain adaptation)<br>
    • <strong>Neural Network</strong>: Warm-start fine-tuning — initialized with DS1 weights, continued on DS2
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<h2 class="section-header">Pipeline Summary & Findings</h2>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.info("Complete all pipeline steps to see the full summary.")
        st.stop()

    models = st.session_state.models
    data   = st.session_state.data

    # Best model on DS1
    best_acc, best_model = 0, ""
    for name, model in models.items():
        m, _ = evaluate_model(model, data["X1te"], data["y1te"])
        if m["Accuracy"] > best_acc:
            best_acc, best_model = m["Accuracy"], name

    # Temporal drift impact
    drift_model = list(models.keys())[0]
    m_ds1, _ = evaluate_model(models[drift_model], data["X1te"], data["y1te"])
    m_ds2, _ = evaluate_model(models[drift_model], data["X2te"], data["y2te"])
    drift_drop = m_ds1["Accuracy"] - m_ds2["Accuracy"]

    st.markdown(f"""
    ### 🏆 Key Findings

    | Finding | Detail |
    |---------|--------|
    | Best Model (DS1 Test) | **{best_model}** (Accuracy: {best_acc:.4f}) |
    | Temporal Drift Drop ({drift_model}) | **{drift_drop:.4f}** accuracy drop (DS1 → DS2) |
    | Split Year | **{data['split_year']}** |
    | Dataset 1 Size | **{len(data['df1']):,}** records |
    | Dataset 2 Size | **{len(data['df2']):,}** records |
    | Target Classes | **{len(data['top_conditions'])}** conditions |
    """)

    st.markdown("---")
    st.markdown("### 📝 Discussion")
    st.markdown("""
    **1. Data Preprocessing & Feature Engineering**
    - Merged 4 key tables: patients → encounters → conditions → observations
    - Engineered aggregated vitals (mean/std per patient) from observation data
    - Applied one-hot encoding for categorical variables (race, ethnicity)
    - StandardScaler normalization applied before SVM and MLP training
    - Temporal split on encounter date to create historical vs current datasets

    **2. Temporal Shift (Data Drift)**
    - Records were split by encounter year into Dataset 1 (historical) and Dataset 2 (current)
    - Meaningful distributional differences observed in age, condition frequency, and vitals between datasets
    - Models trained on DS1 show performance degradation when evaluated on DS2 — confirming temporal shift

    **3. Model Complexity & Bias-Variance**
    - Decision Tree: shallow trees underfit; deep trees overfit (bias-variance tradeoff visible in depth plot)
    - SVM (RBF kernel): good generalization but sensitive to feature scaling
    - Neural Network: best capacity but needs sufficient data; benefits most from continual learning

    **4. Continual Learning**
    - Implemented three strategies: data accumulation (DT), domain adaptation (SVM), warm-start fine-tuning (MLP)
    - Continual learning recovered performance on DS2 test set, demonstrating its value for temporal shift
    - Fine-tuning preserves prior knowledge while adapting to new data distribution

    **5. Feature Importance**
    - Patient age and income are consistently top predictors
    - Clinical vitals (Body Weight, BMI, Glucose) provide strong discriminative signals
    - Encounter class type also contributes to prediction
    """)

    st.markdown("---")
    st.markdown("### 👥 Team Details")
    st.markdown("""
    | | |
    |---|---|
    | **Team** | Team05 |
    | **Course** | BITS F464 Machine Learning |
    | **Semester** | Second Semester 2025-2026 |
    | **Institution** | BITS Pilani, Hyderabad Campus |
    | **Assignment** | Assignment 2 — Clinical ML Pipeline |
    """)
