import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import EmptyDataError, ParserError

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Microplastic Risk Analysis", layout="wide")

NUMERIC_COLS = [
    "MP_Count_per_L",
    "Risk_Score",
    "Microplastic_Size_mm_midpoint",
    "Density_midpoint",
]

CATEGORICAL_COLS = [
    "Location",
    "Shape",
    "Polymer_Type",
    "pH",
    "Salinity",
    "Industrial_Activity",
    "Population_Density",
    "Author",
]

TARGET_RISK_TYPE = "Risk_Type"
TARGET_RISK_LEVEL = "Risk_Level"


# ----------------------------
# DATA LOADING
# ----------------------------
@st.cache_data
def load_data(uploaded_file=None, path: str = "Microplastic.csv") -> pd.DataFrame:
    """Read CSV from upload or local file. Tries common encodings."""
    src = uploaded_file if uploaded_file is not None else path
    last_err = None
    for enc in ("latin1", "utf-8", "cp1252"):
        try:
            return pd.read_csv(src, encoding=enc)
        except (UnicodeDecodeError, EmptyDataError, ParserError) as e:
            last_err = e
        except FileNotFoundError:
            if uploaded_file is None:
                raise
    if last_err:
        raise last_err
    raise ParserError("Could not read CSV.")


# ----------------------------
# PREPROCESSING (EDA-style for your pages)
# ----------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_COLS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.fillna(s.median())

    for c in CATEGORICAL_COLS:
        if c in out.columns:
            mode = out[c].mode(dropna=True)
            if len(mode) > 0:
                out[c] = out[c].fillna(mode.iloc[0])
    return out


def cap_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[c] = np.clip(s, lo, hi)
    return out


def transform_skewed_log1p(df: pd.DataFrame, cols: list[str], threshold: float = 1.0):
    out = df.copy()
    cols_present = [c for c in cols if c in out.columns]
    if not cols_present:
        return out, pd.Series(dtype=float), []

    for c in cols_present:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    skewness = out[cols_present].skew(numeric_only=True)
    skewed_cols = skewness[skewness.abs() > threshold].index.tolist()

    for c in skewed_cols:
        mn = out[c].min()
        if pd.isna(mn):
            continue
        shift = (abs(mn) + 1e-6) if mn <= 0 else 0
        out[c] = np.log1p(out[c] + shift)

    return out, skewness, skewed_cols


def scale_numeric(df: pd.DataFrame, cols: list[str]):
    out = df.copy()
    scaler = StandardScaler()
    cols_present = [c for c in cols if c in out.columns]
    if cols_present:
        out[cols_present] = scaler.fit_transform(out[cols_present])
    return out, scaler


def preprocess_for_model(df_raw: pd.DataFrame):
    """Your EDA-style preprocessing that outputs one-hot encoded matrix X."""
    df = df_raw.copy()

    if TARGET_RISK_TYPE in df.columns and TARGET_RISK_LEVEL in df.columns:
        df = df.dropna(subset=[TARGET_RISK_TYPE, TARGET_RISK_LEVEL])

    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, NUMERIC_COLS)
    df, skewness, skewed_cols = transform_skewed_log1p(df, NUMERIC_COLS)
    df, _ = scale_numeric(df, NUMERIC_COLS)

    y_type = df[TARGET_RISK_TYPE] if TARGET_RISK_TYPE in df.columns else None
    y_level = df[TARGET_RISK_LEVEL] if TARGET_RISK_LEVEL in df.columns else None

    drop_targets = [c for c in (TARGET_RISK_TYPE, TARGET_RISK_LEVEL) if c in df.columns]
    X_base = df.drop(columns=drop_targets)

    cat_present = [c for c in CATEGORICAL_COLS if c in X_base.columns]
    X = pd.get_dummies(X_base, columns=cat_present, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df, X, y_type, y_level, skewness, skewed_cols


# ----------------------------
# SPLIT HELPERS
# ----------------------------
def merge_rare_classes(y: pd.Series, min_count: int = 2, other_label: str = "Other") -> pd.Series:
    y = pd.Series(y).copy()
    counts = y.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    return y.where(~y.isin(rare), other_label)


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    y = pd.Series(y)
    m = y.notna()
    X, y = X.loc[m], y.loc[m]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target.")

    counts = y.value_counts()
    k = y.nunique()
    n = len(y)
    min_class = int(counts.min())

    if min_class < 2:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_tr, X_te, y_tr, y_te), False, float(test_size)

    min_ts = k / n
    max_ts = 1 - k / n
    ts = min(max(float(test_size), min_ts), max_ts if max_ts > 0 else float(test_size))

    for ts_try in (ts, 0.2, 0.15, 0.1, 0.05):
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=ts_try, random_state=random_state, stratify=y
            )
            return (X_tr, X_te, y_tr, y_te), True, float(ts_try)
        except ValueError:
            continue

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_tr, X_te, y_tr, y_te), False, float(test_size)


# ----------------------------
# MODELING (holdout split)
# ----------------------------
def train_models_holdout(X, y, test_size=0.2):
    y = pd.Series(y)
    m = y.notna()
    X, y = X.loc[m], y.loc[m]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes to train.")

    before = y.value_counts()
    y = merge_rare_classes(y, min_count=2, other_label="Other")
    after = y.value_counts()
    merge_note = None if before.equals(after) else {"before": before, "after": after}

    (X_tr, X_te, y_tr, y_te), used_strat, final_ts = safe_train_test_split(X, y, test_size=test_size)

    split_note = (
        f"✅ Stratified split used (test_size={final_ts:.2f})."
        if used_strat
        else f"⚠️ Non-stratified split used (test_size={final_ts:.2f}) due to small classes."
    )
    split_info = {
        "X_train_shape": X_tr.shape,
        "X_test_shape": X_te.shape,
        "y_train_counts": y_tr.value_counts(),
        "y_test_counts": y_te.value_counts(),
        "used_stratify": used_strat,
        "final_test_size": final_ts,
    }

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision (weighted)": precision_score(y_te, pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_te, pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_te, pred, average="weighted", zero_division=0),
        })

    return models, pd.DataFrame(rows).set_index("Model"), split_info, split_note, merge_note


def smote_and_tune_lr_holdout(X, y, test_size=0.2):
    y = pd.Series(y)
    m = y.notna()
    X, y = X.loc[m], y.loc[m]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes.")

    before = y.value_counts()
    y = merge_rare_classes(y, min_count=2, other_label="Other")
    after = y.value_counts()
    merge_note = None if before.equals(after) else {"before": before, "after": after}

    (X_tr, X_te, y_tr, y_te), used_strat, final_ts = safe_train_test_split(X, y, test_size=test_size)

    split_note = (
        f"✅ Stratified split used (test_size={final_ts:.2f})."
        if used_strat
        else f"⚠️ Non-stratified split used (test_size={final_ts:.2f}) due to small classes."
    )
    split_info = {
        "X_train_shape": X_tr.shape,
        "X_test_shape": X_te.shape,
        "y_train_counts": y_tr.value_counts(),
        "y_test_counts": y_te.value_counts(),
        "used_stratify": used_strat,
        "final_test_size": final_ts,
    }

    smote_used = True
    try:
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    except ValueError:
        smote_used = False
        X_res, y_res = X_tr, y_tr

    grid = GridSearchCV(
        LogisticRegression(max_iter=1500, multi_class="auto"),
        param_grid={"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_res, y_res)

    best = grid.best_estimator_
    pred = best.predict(X_te)

    tuned_df = pd.DataFrame([{
        "Model": "LogReg (tuned + SMOTE)" if smote_used else "LogReg (tuned, no SMOTE)",
        "Accuracy": accuracy_score(y_te, pred),
        "Precision (weighted)": precision_score(y_te, pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_te, pred, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_te, pred, average="weighted", zero_division=0),
    }]).set_index("Model")

    return best, tuned_df, grid.best_params_, split_info, split_note, merge_note, smote_used


# ----------------------------
# CV (leakage-safe) helpers
# ----------------------------
def build_cv_preprocessor(df_raw: pd.DataFrame) -> ColumnTransformer:
    """Preprocessing done inside folds (leakage-safe)."""
    numeric = [c for c in NUMERIC_COLS if c in df_raw.columns]
    categorical = [c for c in CATEGORICAL_COLS if c in df_raw.columns]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
    )


def run_cv(df_raw: pd.DataFrame, target_col: str, model_key: str,
           k: int = 5, stratified: bool = True, use_smote: bool = False):
    if target_col not in df_raw.columns:
        raise ValueError(f"Missing target column: {target_col}")

    df = df_raw.dropna(subset=[target_col]).copy()
    y = merge_rare_classes(df[target_col], min_count=2, other_label="Other")

    # X = raw features only (drop targets)
    X = df.drop(columns=[c for c in (TARGET_RISK_TYPE, TARGET_RISK_LEVEL) if c in df.columns])

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in target for CV.")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    if model_key not in models:
        raise ValueError("Unknown model choice.")
    model = models[model_key]

    splitter = (
        StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        if stratified else
        KFold(n_splits=k, shuffle=True, random_state=42)
    )

    prep = build_cv_preprocessor(df_raw)

    if use_smote:
        pipe = ImbPipeline(steps=[
            ("prep", prep),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])
    else:
        pipe = Pipeline(steps=[
            ("prep", prep),
            ("model", model),
        ])

    scoring = {
        "accuracy": "accuracy",
        "precision_w": "precision_weighted",
        "recall_w": "recall_weighted",
        "f1_w": "f1_weighted",
    }

    scores = cross_validate(pipe, X, y, cv=splitter, scoring=scoring, n_jobs=-1, error_score="raise")

    summary = {}
    for key in scoring:
        vals = scores[f"test_{key}"]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary_df = pd.DataFrame(summary).T.rename(index={
        "accuracy": "Accuracy",
        "precision_w": "Precision (weighted)",
        "recall_w": "Recall (weighted)",
        "f1_w": "F1-score (weighted)"
    })

    return summary_df, scores


# ----------------------------
# PLOTS
# ----------------------------
def plot_hist_box(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if s.empty:
        axes[0].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
        axes[1].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
    else:
        sns.histplot(s, kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=s, ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
    plt.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = x.notna() & y.notna()
    fig, ax = plt.subplots(figsize=(6, 4))
    if m.sum() == 0:
        ax.text(0.5, 0.5, f"No numeric data for {x_col} and {y_col}", ha="center", va="center")
    else:
        ax.scatter(x[m], y[m], alpha=0.7)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics_df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df[["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]].plot(kind="bar", ax=ax)
    ax.set_title(f"Model Performance {title_suffix}")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_box_by_category(df, value_col, category_col, top_n=8, other_label="Other", figsize=(12, 5)):
    val = pd.to_numeric(df[value_col], errors="coerce")
    cat = df[category_col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    data = pd.DataFrame({value_col: val, category_col: cat}).dropna()

    fig, ax = plt.subplots(figsize=figsize)
    if data.empty:
        ax.text(0.5, 0.5, "No usable data", ha="center", va="center")
        plt.tight_layout()
        return fig

    counts = data[category_col].value_counts()
    keep = counts.head(top_n).index
    data[category_col] = np.where(data[category_col].isin(keep), data[category_col], other_label)

    order = data.groupby(category_col)[value_col].median().sort_values().index.tolist()
    sns.boxplot(data=data, y=category_col, x=value_col, order=order, ax=ax)
    ax.set_title(f"{value_col} by {category_col} (Top {top_n} + {other_label})")
    plt.tight_layout()
    return fig


def plot_cat_topn(series: pd.Series, title: str, top_n=15, other_label="Other", figsize=(10, 6)):
    s = series.dropna().astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
    counts = s.value_counts()
    fig, ax = plt.subplots(figsize=figsize)

    if counts.empty:
        ax.text(0.5, 0.5, "No category data available", ha="center", va="center")
        plt.tight_layout()
        return fig, counts

    top = counts.head(top_n)
    rem = counts.iloc[top_n:].sum()
    if rem > 0:
        top = pd.concat([top, pd.Series({other_label: rem})])

    top.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    plt.tight_layout()
    return fig, counts


def plot_importances(imp: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    imp.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ----------------------------
# APP
# ----------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        "Explore microplastic risk data, preprocessing, modeling, **and** leakage-safe cross-validation."
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Data Overview & Task 1",
            "Preprocessing (Task 2)",
            "Feature Selection & Relevance (Task 3 & 6)",
            "Classification Modeling (Tasks 4, 5 & 7)",
            "Polymer Type Distribution",
            "SMOTE & Hyperparameter Tuning (Risk_Type)",
            "Cross Validation (K-Fold)",
        ],
    )

    st.sidebar.subheader("Data source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Microplastic CSV",
        type=["csv"],
        help="If you don't upload anything, the app will try to use 'Microplastic.csv' from the app folder.",
    )

    try:
        df_raw = load_data(uploaded_file=uploaded_file)
    except UnicodeDecodeError:
        st.error("⚠️ Unable to decode the file. Please upload a proper CSV (text).")
        st.stop()
    except EmptyDataError:
        st.error("⚠️ The uploaded file appears empty/unreadable as CSV.")
        st.stop()
    except ParserError:
        st.error("⚠️ The file is not a valid CSV format. Re-export as CSV and try again.")
        st.stop()
    except FileNotFoundError:
        st.error("❌ No dataset found. Upload a CSV or add 'Microplastic.csv' beside app.py.")
        st.stop()

    # -------- PAGE 1
    if page == "Data Overview & Task 1":
        st.header("Data Overview & Task 1: Risk_Score Analysis")
        t1, t2, t3, t4 = st.tabs(["Raw Data", "Risk_Score Distribution", "MP_Count vs Risk_Score", "Risk_Score by Risk_Level"])

        with t1:
            st.dataframe(df_raw.head(10))
            st.markdown(f"**Shape:** `{df_raw.shape[0]}` rows × `{df_raw.shape[1]}` columns")

        with t2:
            if "Risk_Score" in df_raw.columns:
                st.pyplot(plot_hist_box(df_raw, "Risk_Score"))
            else:
                st.info("Column 'Risk_Score' not found.")

        with t3:
            if {"MP_Count_per_L", "Risk_Score"}.issubset(df_raw.columns):
                st.pyplot(plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score"))
            else:
                st.info("Columns 'MP_Count_per_L' and/or 'Risk_Score' not found.")

        with t4:
            if {"Risk_Level", "Risk_Score"}.issubset(df_raw.columns):
                st.pyplot(plot_box_by_category(df_raw, "Risk_Score", "Risk_Level"))
            else:
                st.info("Columns 'Risk_Level' and/or 'Risk_Score' not found.")

    # -------- PAGE 2
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Preprocessing")
        df_clean, X, y_type, y_level, skewness, skewed_cols = preprocess_for_model(df_raw)

        t1, t2, t3, t4 = st.tabs(["Before", "After", "Skewness", "Encoded X"])
        with t1:
            present = [c for c in NUMERIC_COLS if c in df_raw.columns]
            st.write(df_raw[present].describe() if present else "No numeric cols present.")
        with t2:
            present = [c for c in NUMERIC_COLS if c in df_clean.columns]
            st.write(df_clean[present].describe() if present else "No numeric cols present.")
        with t3:
            st.write(skewness)
            st.write("Transformed:", skewed_cols if skewed_cols else "None")
        with t4:
            st.dataframe(X.head(10))
            st.write("X shape:", X.shape)

    # -------- PAGE 3
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance")
        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)
        rt, rl = st.tabs(["Risk_Type", "Risk_Level"])

        with rt:
            if y_type is None:
                st.warning("Risk_Type not found.")
            else:
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(X, y_type)
                imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.dataframe(imp.head(10))
                st.pyplot(plot_importances(imp.head(10), "Top 10 Feature Importances (Risk_Type)"))

        with rl:
            if y_level is None:
                st.warning("Risk_Level not found.")
            else:
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(X, y_level)
                imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.dataframe(imp.head(10))
                st.pyplot(plot_importances(imp.head(10), "Top 10 Feature Importances (Risk_Level)"))

    # -------- PAGE 4
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Tasks 4, 5 & 7: Classification Modeling")
        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        rt, rl = st.tabs(["Risk-Type Models", "Risk-Level Models"])
        with rt:
            if y_type is None:
                st.warning("Risk_Type not found.")
            else:
                _, mdf, split_info, split_note, merge_note = train_models_holdout(X, y_type)
                st.dataframe(mdf.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(mdf, "(Risk-Type)"))
                st.info(split_note)
                if merge_note:
                    with st.expander("Rare-class merging details"):
                        st.write(merge_note["before"])
                        st.write(merge_note["after"])
                st.write("Train counts:", split_info["y_train_counts"])
                st.write("Test counts:", split_info["y_test_counts"])

        with rl:
            if y_level is None:
                st.warning("Risk_Level not found.")
            else:
                _, mdf, split_info, split_note, merge_note = train_models_holdout(X, y_level)
                st.dataframe(mdf.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(mdf, "(Risk-Level)"))
                st.info(split_note)
                if merge_note:
                    with st.expander("Rare-class merging details"):
                        st.write(merge_note["before"])
                        st.write(merge_note["after"])
                st.write("Train counts:", split_info["y_train_counts"])
                st.write("Test counts:", split_info["y_test_counts"])

    # -------- PAGE 5
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution")
        df = handle_missing_values(df_raw)
        if "Polymer_Type" not in df.columns:
            st.warning("Column 'Polymer_Type' not found.")
        else:
            tabA, tabB = st.tabs(["Counts Table", "Readable Plot (Top N + Other)"])
            polymer = df["Polymer_Type"]

            with tabA:
                vc = polymer.dropna().astype(str).str.strip().replace({"": np.nan}).dropna().value_counts()
                st.dataframe(vc.rename("count"))

            with tabB:
                top_n = st.slider("Show Top N polymer types", 5, 30, 15, 1)
                fig, _ = plot_cat_topn(polymer, f"Distribution of Polymer_Type (Top {top_n} + Other)", top_n=top_n)
                st.pyplot(fig)

    # -------- PAGE 6
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("SMOTE & Hyperparameter Tuning (Risk_Type)")
        _, X, y_type, _, _, _ = preprocess_for_model(df_raw)

        if y_type is None:
            st.warning("Risk_Type not found.")
            return

        a, b = st.tabs(["Base Models", "SMOTE + Tuned Logistic Regression"])
        with a:
            st.write(pd.Series(y_type).value_counts())
            _, base_mdf, _, split_note, merge_note = train_models_holdout(X, y_type)
            st.dataframe(base_mdf.style.format("{:.3f}"))
            st.info(split_note)
            if merge_note:
                with st.expander("Rare-class merging details"):
                    st.write(merge_note["before"])
                    st.write(merge_note["after"])

        with b:
            with st.spinner("Running SMOTE + GridSearchCV..."):
                _, tuned_mdf, best_params, _, split_note, merge_note, smote_used = smote_and_tune_lr_holdout(X, y_type)
            st.json(best_params)
            st.info(split_note)
            if not smote_used:
                st.warning("SMOTE could not be applied; tuning continued without SMOTE.")
            if merge_note:
                with st.expander("Rare-class merging details"):
                    st.write(merge_note["before"])
                    st.write(merge_note["after"])
            st.dataframe(tuned_mdf.style.format("{:.3f}"))

    # -------- PAGE 7 (NEW)
    elif page == "Cross Validation (K-Fold)":
        st.header("Cross Validation (K-Fold / Stratified K-Fold)")
        st.markdown(
            """
            This page runs **leakage-safe cross-validation** using a **Pipeline**.
            - Preprocessing happens inside folds.
            - Optional SMOTE (if enabled) happens inside folds as well.
            """
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            target = st.selectbox("Target", [TARGET_RISK_TYPE, TARGET_RISK_LEVEL])
            model_key = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
            k = st.slider("Number of folds (k)", 3, 10, 5, 1)
            stratified = st.checkbox("Use Stratified K-Fold (recommended)", value=True)
            use_smote = st.checkbox("Use SMOTE (for imbalanced targets)", value=False)

            st.subheader("Target distribution (after rare-class merge)")
            if target in df_raw.columns:
                y_prev = merge_rare_classes(df_raw[target].dropna(), min_count=2, other_label="Other")
                st.write(y_prev.value_counts())
            else:
                st.warning(f"Missing column: {target}")

        with col2:
            st.subheader("Run CV")
            if st.button("Run Cross-Validation", type="primary"):
                try:
                    with st.spinner("Running cross-validation..."):
                        summary_df, raw_scores = run_cv(
                            df_raw=df_raw,
                            target_col=target,
                            model_key=model_key,
                            k=k,
                            stratified=stratified,
                            use_smote=use_smote,
                        )

                    st.success("Done!")
                    show = summary_df.copy()
                    show["mean±std"] = show.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
                    st.dataframe(show[["mean±std"]])

                    with st.expander("Per-fold scores"):
                        fold_df = pd.DataFrame({
                            "Accuracy": raw_scores["test_accuracy"],
                            "Precision (weighted)": raw_scores["test_precision_w"],
                            "Recall (weighted)": raw_scores["test_recall_w"],
                            "F1-score (weighted)": raw_scores["test_f1_w"],
                        })
                        fold_df.index = [f"Fold {i+1}" for i in range(len(fold_df))]
                        st.dataframe(fold_df.style.format("{:.3f}"))

                except Exception as e:
                    st.error(f"Cross-validation failed: {e}")


if __name__ == "__main__":
    main()
