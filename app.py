"""
Streamlit app for MicroPlastic.csv analysis and classification.

Save as app.py and run with:
    pip install -r requirements.txt
    streamlit run app.py

Requirements (suggested):
pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost streamlit
"""
import warnings
warnings.filterwarnings("ignore")

import io
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# XGBoost is optional (wrapped import)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

st.set_page_config(layout="wide", page_title="Microplastic Risk Analysis")

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_COLS = [
    "Location",
    "Shape",
    "Polymer_Type",
    "pH",
    "Salinity",
    "Industrial_Activity",
    "Population_Density",
    "Risk_Type",
    "Risk_Level",
    "Author",
]
NUMERIC_COLS = [
    "MP_Count_per_L",
    "Risk_Score",
    "Microplastic_Size_mm_midpoint",
    "Density_midpoint",
]


@st.cache_data(show_spinner=False)
def load_dataframe(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def initial_inspect(df: pd.DataFrame):
    st.subheader("Data preview & info")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(10))
    st.write("Missing values per column:")
    st.write(df.isna().sum())


def plot_polymer_distribution(df: pd.DataFrame):
    if "Polymer_Type" not in df.columns:
        st.info("Polymer_Type column not found.")
        return
    st.subheader("Polymer_Type distribution")
    missing_before = df["Polymer_Type"].isna().sum()
    st.write(f"Missing Polymer_Type before fill: {missing_before}")
    df["Polymer_Type"] = df["Polymer_Type"].fillna("Unknown")
    missing_after = df["Polymer_Type"].isna().sum()
    st.write(f"Missing Polymer_Type after fill: {missing_after}")
    plt.figure(figsize=(6, 4))
    order = df["Polymer_Type"].value_counts().index
    sns.countplot(data=df, y="Polymer_Type", order=order)
    plt.title("Polymer_Type counts")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_risk_score_distribution(df: pd.DataFrame):
    if "Risk_Score" not in df.columns:
        return
    st.subheader("Risk_Score distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Risk_Score"].dropna(), kde=True, ax=ax)
    st.pyplot(fig)
    plt.clf()


def visualize_risk_score_by_level(df: pd.DataFrame):
    if ("Risk_Score" in df.columns) and ("Risk_Level" in df.columns):
        st.subheader("Risk_Score by Risk_Level (violin & box)")
        fig, ax = plt.subplots(figsize=(8, 4))
        order = sorted(df["Risk_Level"].dropna().unique())
        sns.violinplot(x="Risk_Level", y="Risk_Score", data=df, inner="quartile", order=order, ax=ax)
        st.pyplot(fig)
        plt.clf()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="Risk_Level", y="Risk_Score", data=df, order=order, ax=ax2)
        st.pyplot(fig2)
        plt.clf()
    else:
        st.info("Risk_Score and/or Risk_Level not available for grouped visualizations.")


def scatter_risk_score_vs_mp_count(df: pd.DataFrame):
    if ("Risk_Score" in df.columns) and ("MP_Count_per_L" in df.columns):
        st.subheader("Risk_Score vs MP_Count_per_L")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="MP_Count_per_L", y="Risk_Score", data=df, alpha=0.6, ax=ax)
        sns.regplot(x="MP_Count_per_L", y="Risk_Score", data=df, scatter=False, color="red", lowess=True, ax=ax)
        st.pyplot(fig)
        plt.clf()


def detect_and_cap_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    caps = {}
    st.subheader("Outliers detection & capping (IQR method)")
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_lower = (df[col] < lower).sum()
        n_upper = (df[col] > upper).sum()
        st.write(f"{col}: {n_lower} lower outliers, {n_upper} upper outliers. Capping to [{lower:.3g}, {upper:.3g}]")
        df[col] = df[col].clip(lower=lower, upper=upper)
        caps[col] = (lower, upper)
    return caps


def transform_skewed_columns(df: pd.DataFrame, numeric_cols: List[str], skew_threshold=0.75) -> List[str]:
    transformed = []
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    st.subheader("Skewness & transformations")
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dropna().dtype, np.number):
            continue
        skew = df[col].dropna().skew()
        st.write(f"Skewness of {col}: {skew:.3f}")
        if abs(skew) > skew_threshold:
            non_na_idx = df[col].notna()
            try:
                vals = df.loc[non_na_idx, col].values.reshape(-1, 1)
                transformed_vals = pt.fit_transform(vals).flatten()
                df.loc[non_na_idx, col] = transformed_vals
                transformed.append(col)
                st.write(f"Applied Yeo-Johnson transform to {col}")
            except Exception:
                if (df[col] >= 0).all():
                    df[col] = np.log1p(df[col])
                    transformed.append(col)
                    st.write(f"Applied log1p transform to {col} (fallback)")
                else:
                    st.write(f"Could not transform {col} (skipped)")
    return transformed


def build_preprocessor(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    low_card_cols = [c for c in categorical_cols if c in df.columns and df[c].nunique() <= 20]
    high_card_cols = [c for c in categorical_cols if c in df.columns and c not in low_card_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_low_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    categorical_high_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if any(c in df.columns for c in numeric_cols):
        transformers.append(("num", numeric_transformer, [c for c in numeric_cols if c in df.columns]))
    if low_card_cols:
        transformers.append(("cat_low", categorical_low_transformer, low_card_cols))
    if high_card_cols:
        transformers.append(("cat_high", categorical_high_transformer, high_card_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, df: pd.DataFrame) -> List[str]:
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if isinstance(trans, Pipeline):
            last_step = trans.steps[-1][1]
        else:
            last_step = trans
        try:
            names = last_step.get_feature_names_out(cols)
            feature_names.extend(list(names))
        except Exception:
            # fallback: use the input column names
            feature_names.extend(cols)
    return feature_names


def feature_selection_ranking(X: np.ndarray, y: pd.Series, feature_names: List[str], k=20) -> pd.Series:
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    return mi_series


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, apply_smote=False, target_name="target"):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE)

    if apply_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        bal = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "f1_macro": f1, "balanced_accuracy": bal, "precision": prec, "recall": rec}
        trained_models[name] = model
    return results, trained_models


def map_feature_importances(model, feature_names: List[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            coefs = np.abs(coef)
        else:
            coefs = np.mean(np.abs(coef), axis=0)
        return pd.Series(coefs, index=feature_names).sort_values(ascending=False)
    else:
        return pd.Series(dtype=float)


# Streamlit layout
st.title("Microplastic Risk Analysis & Classification")
st.markdown("Upload a dataset (CSV or Excel). The app will preprocess, visualize, and run classification models to predict Risk_Type or Risk_Level.")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

df = load_dataframe(uploaded_file)

if df is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

# Show initial inspection
initial_inspect(df)

# Polymer_Type handling & distribution
plot_polymer_distribution(df)

# Visualizations
plot_risk_score_distribution(df)
visualize_risk_score_by_level(df)
scatter_risk_score_vs_mp_count(df)

# Outlier handling controls
st.subheader("Preprocessing controls")
do_outlier = st.checkbox("Detect and cap outliers in numeric columns (IQR)", value=True)
do_transform = st.checkbox("Transform skewed numeric columns (Yeo-Johnson/log1p)", value=True)

working_df = df.copy()

if do_outlier:
    caps = detect_and_cap_outliers(working_df, NUMERIC_COLS)
else:
    caps = {}

if do_transform:
    transformed_cols = transform_skewed_columns(working_df, NUMERIC_COLS, skew_threshold=0.75)
    if transformed_cols:
        st.success(f"Transformed columns: {transformed_cols}")
else:
    transformed_cols = []

st.write("Columns considered numeric:", [c for c in NUMERIC_COLS if c in working_df.columns])
st.write("Columns considered categorical:", [c for c in CATEGORICAL_COLS if c in working_df.columns])

# Modeling section
st.subheader("Modeling")
target_option = st.selectbox("Select target to predict", options=[c for c in ["Risk_Type", "Risk_Level"] if c in working_df.columns])
use_smote = st.checkbox("Apply SMOTE to training data (recommended for Risk_Type)", value=True)
run_modeling = st.button("Run modeling pipeline")

if run_modeling:
    st.info(f"Running modeling for target: {target_option}")
    target_vars = ["Risk_Type", "Risk_Level"]
    drop_cols = [c for c in target_vars if c in working_df.columns]
    X = working_df.drop(columns=drop_cols)
    y = working_df[target_option].copy()
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    if np.issubdtype(y.dtype, np.number):
        y = y.astype(str)

    # show value counts for Risk_Type
    if target_option == "Risk_Type":
        st.write("Value counts for Risk_Type (full):")
        st.write(df["Risk_Type"].value_counts())
        st.write("Value counts for Risk_Type (filtered for modeling):")
        st.write(y.value_counts())

    # Build preprocessor and fit
    preprocessor = build_preprocessor(X, NUMERIC_COLS, [c for c in CATEGORICAL_COLS if c in X.columns])
    preprocessor.fit(X)
    X_t = preprocessor.transform(X)

    feature_names = get_feature_names(preprocessor, X)
    st.write(f"Number of features after preprocessing: {len(feature_names)}")

    # Feature ranking
    try:
        mi = feature_selection_ranking(X_t, y, feature_names, k=20)
        st.subheader("Top features by mutual information")
        st.table(mi.head(20))
        top_features = mi.index[:20].tolist()
    except Exception as e:
        st.warning(f"Feature ranking failed: {e}")
        top_features = feature_names[:20]

    # Select top features for modeling
    selected_idx = [i for i, f in enumerate(feature_names) if f in top_features]
    if selected_idx:
        X_sel = X_t[:, selected_idx]
        sel_feature_names = [feature_names[i] for i in selected_idx]
    else:
        X_sel = X_t
        sel_feature_names = feature_names

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Apply SMOTE if requested
    apply_sm = use_smote and (target_option == "Risk_Type")
    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test, sel_feature_names, apply_smote=apply_sm, target_name=target_option)

    st.subheader("Model performance")
    perf_df = pd.DataFrame(results).T
    st.dataframe(perf_df.style.format("{:.4f}"))

    # Tune logistic regression for Risk_Type as requested
    if target_option == "Risk_Type":
        st.subheader("Tuning Logistic Regression (GridSearchCV)")
        try:
            lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
            param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "saga"]}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            # For tuning use SMOTE-resampled training if requested
            if apply_sm:
                sm = SMOTE(random_state=RANDOM_STATE)
                X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
            else:
                X_train_sm, y_train_sm = X_train, y_train
            grid = GridSearchCV(lr, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
            grid.fit(X_train_sm, y_train_sm)
            best_lr = grid.best_estimator_
            st.write("Best params:", grid.best_params_)
            y_pred = best_lr.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            bal = balanced_accuracy_score(y_test, y_pred)
            st.write("Tuned LogisticRegression performance:")
            st.write({"accuracy": acc, "f1_macro": f1, "balanced_accuracy": bal})
            results["LogisticRegression_Tuned"] = {"accuracy": acc, "f1_macro": f1, "balanced_accuracy": bal}
            trained_models["LogisticRegression_Tuned"] = best_lr
        except Exception as e:
            st.error(f"Hyperparameter tuning failed: {e}")

    # Feature importances
    st.subheader("Feature relevance (top 15 per model)")
    for name, model in trained_models.items():
        fi = map_feature_importances(model, sel_feature_names)
        if fi.empty:
            st.write(f"{name}: no feature importance available")
            continue
        st.write(f"{name} top features")
        st.table(fi.head(15))

    # Model comparison bar chart
    st.subheader("Model comparison (F1_macro vs Accuracy)")
    fig, ax = plt.subplots(figsize=(6, 3))
    models_plot = list(results.keys())
    f1s = [results[m]["f1_macro"] for m in models_plot]
    accs = [results[m]["accuracy"] for m in models_plot]
    x = np.arange(len(models_plot))
    ax.bar(x - 0.15, f1s, width=0.3, label="F1_macro")
    ax.bar(x + 0.15, accs, width=0.3, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models_plot, rotation=45)
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)
    plt.clf()

    st.success("Modeling complete. See results and plots above.")

st.markdown("App finished. Upload another dataset or change options to rerun.")
