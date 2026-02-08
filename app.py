import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import EmptyDataError, ParserError

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif, chi2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._encoders")

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
TARGET_RISK_TYPE = "Risk_Type"
TARGET_RISK_LEVEL = "Risk_Level"

NUMERIC_COLS = [
    "MP_Count_per_L",
    "Risk_Score",
    "Microplastic_Size_mm",
    "Density",
    "Latitude",
    "Longitude",
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
    "Source",
]

DEFAULT_MODEL_DROP_COLS = ["Location", "Author"]


# -------------------------------------------------------
# LOADING + CLEANING
# -------------------------------------------------------
def load_data(uploaded_file=None):
    """Robust CSV reader with encoding fallbacks."""
    if uploaded_file is None:
        path = "Microplastic.csv"
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, sep=None, engine="python")
    else:
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=None, engine="python")


def handle_missing_values(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df


def encode_categorical_columns(df: pd.DataFrame, categorical_cols: list, drop_cols_for_model: tuple = ()):
    """Encode categorical columns with proper reporting."""
    df_encoded = df.copy()
    encoding_report = []
    
    for col in categorical_cols:
        if col not in df_encoded.columns or col in drop_cols_for_model:
            continue
        
        if col in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL]:
            unique_vals = df_encoded[col].dropna().unique()
            label_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            df_encoded[col] = df_encoded[col].map(label_map)
            
            encoding_report.append({
                "Column": col,
                "Encoding_Type": "Label Encoding",
                "Unique_Values": len(unique_vals),
                "Values": str(list(unique_vals)[:5]) + ("..." if len(unique_vals) > 5 else ""),
                "Result_Columns": col,
            })
        else:
            unique_vals = df_encoded[col].dropna().unique()
            encoding_report.append({
                "Column": col,
                "Encoding_Type": "One-Hot Encoding",
                "Unique_Values": len(unique_vals),
                "Values": str(list(unique_vals)[:5]) + ("..." if len(unique_vals) > 5 else ""),
                "Result_Columns": f"{len(unique_vals)} binary columns",
            })
    
    return df_encoded, pd.DataFrame(encoding_report)


def detect_and_handle_outliers(df: pd.DataFrame, numeric_cols: list):
    """Detect and handle outliers using IQR method."""
    df_outliers_handled = df.copy()
    outlier_report = []
    
    for col in numeric_cols:
        if col not in df_outliers_handled.columns:
            continue
        
        s = pd.to_numeric(df_outliers_handled[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_count = ((s < lower_bound) | (s > upper_bound)).sum()
        
        outlier_report.append({
            "Column": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower_Bound": round(lower_bound, 4),
            "Upper_Bound": round(upper_bound, 4),
            "Outliers_Detected": int(outliers_count),
            "Outlier_Percentage": round((outliers_count / s.notna().sum() * 100), 2),
        })
        
        df_outliers_handled[col] = s.clip(lower_bound, upper_bound)
    
    return df_outliers_handled, pd.DataFrame(outlier_report)


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = s.clip(lower, upper)
    return df


def transform_skewed(df: pd.DataFrame, numeric_cols, threshold=0.5):
    """Transform skewed numerical columns and return analysis."""
    df = df.copy()
    present = [c for c in numeric_cols if c in df.columns]
    
    skewness_before = df[present].apply(lambda x: pd.to_numeric(x, errors="coerce")).skew(numeric_only=True)
    skewed_cols = skewness_before[skewness_before.abs() > threshold].index.tolist()
    
    transformation_report = []
    df_transformed = df.copy()
    
    for col in skewed_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        
        shift = -s.min() if s.min() < 0 else 0
        s_log = np.log1p(s + shift)
        skewness_after = s_log.skew()
        
        transformation_report.append({
            "Column": col,
            "Skewness_Before": round(skewness_before[col], 4),
            "Skewness_After": round(skewness_after, 4),
            "Skewness_Reduced": round(abs(skewness_before[col]) - abs(skewness_after), 4),
            "Transformation": "Log1p",
            "Shift_Applied": shift,
        })
        
        df_transformed[col] = s_log
    
    return df_transformed, skewness_before, skewed_cols, pd.DataFrame(transformation_report)


def scale_numeric_with_report(df: pd.DataFrame, numeric_cols):
    """Apply StandardScaler and return scaling report."""
    df = df.copy()
    scaler = StandardScaler()
    present = [c for c in numeric_cols if c in df.columns]
    
    scaling_report = []
    
    if present:
        vals = df[present].apply(pd.to_numeric, errors="coerce")
        original_stats = vals.describe().T
        
        scaled_vals = scaler.fit_transform(vals.fillna(vals.median()))
        df[present] = scaled_vals
        
        scaled_stats = pd.DataFrame(scaled_vals, columns=present).describe().T
        
        for col in present:
            orig_mean = original_stats.loc[col, "mean"]
            orig_std = original_stats.loc[col, "std"]
            scaled_mean = scaled_stats.loc[col, "mean"]
            scaled_std = scaled_stats.loc[col, "std"]
            
            scaling_report.append({
                "Column": col,
                "Original_Mean": round(orig_mean, 4),
                "Original_Std": round(orig_std, 4),
                "Scaled_Mean": round(scaled_mean, 6),
                "Scaled_Std": round(scaled_std, 4),
                "Scaling_Method": "StandardScaler (z-score)",
            })
    
    return df, scaler, pd.DataFrame(scaling_report)


def coerce_numeric_like(df: pd.DataFrame, columns):
    df = df.copy()
    for c in columns:
        if c in df.columns:
            s = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


# -------------------------------------------------------
# FEATURE SELECTION METHODS
# -------------------------------------------------------
def compute_mutual_information(X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> pd.DataFrame:
    """Compute Mutual Information scores."""
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            "Feature": X.columns,
            "MI_Score": mi_scores,
        }).sort_values("MI_Score", ascending=False)
        return mi_df.head(n_top)
    except Exception as e:
        st.error(f"MI computation failed: {e}")
        return pd.DataFrame()


def compute_chi_squared(X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> pd.DataFrame:
    """Compute Chi-squared scores."""
    try:
        X_shifted = X - X.min() + 1e-10
        chi2_scores = chi2(X_shifted, y)[0]
        
        chi2_df = pd.DataFrame({
            "Feature": X.columns,
            "Chi2_Score": chi2_scores,
        }).sort_values("Chi2_Score", ascending=False)
        return chi2_df.head(n_top)
    except Exception as e:
        st.warning(f"Chi-squared computation failed: {e}")
        return pd.DataFrame()


def compute_rf_importance(X: pd.DataFrame, y: pd.Series, n_top: int = 20, n_estimators: int = 200) -> pd.DataFrame:
    """Compute Random Forest feature importance."""
    try:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        rf_df = pd.DataFrame({
            "Feature": X.columns,
            "RF_Importance": rf.feature_importances_,
        }).sort_values("RF_Importance", ascending=False)
        return rf_df.head(n_top)
    except Exception as e:
        st.error(f"RF importance computation failed: {e}")
        return pd.DataFrame()


def rank_features_multi_method(X: pd.DataFrame, y: pd.Series, n_top: int = 15) -> dict:
    """Rank features using multiple methods."""
    results = {}
    
    st.info("Computing Mutual Information scores...")
    mi_scores = compute_mutual_information(X, y, n_top=n_top)
    results["Mutual Information"] = mi_scores
    
    st.info("Computing Chi-squared scores...")
    chi2_scores = compute_chi_squared(X, y, n_top=n_top)
    results["Chi-squared"] = chi2_scores
    
    st.info("Computing Random Forest feature importance...")
    rf_scores = compute_rf_importance(X, y, n_top=n_top, n_estimators=200)
    results["Random Forest"] = rf_scores
    
    return results


# -------------------------------------------------------
# SPLIT HELPERS
# -------------------------------------------------------
def merge_rare_classes(y: pd.Series, min_count: int = 2, other_label: str = "Other"):
    y = pd.Series(y).copy()
    counts = y.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    y = y.where(~y.isin(rare), other_label)
    return y


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target.")

    counts = y.value_counts()
    min_class = int(counts.min())
    n = len(y)
    k = y.nunique()

    if min_class < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        return (X_train, X_test, y_train, y_test), False, float(test_size)

    min_test_size = k / n
    max_test_size = 1 - (k / n)

    ts = float(test_size)
    ts = max(ts, min_test_size)
    if max_test_size > 0:
        ts = min(ts, max_test_size)

    for ts_try in [ts, 0.2, 0.15, 0.1, 0.05]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts_try, random_state=random_state, stratify=y
            )
            return (X_train, X_test, y_train, y_test), True, float(ts_try)
        except ValueError:
            continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    return (X_train, X_test, y_train, y_test), False, float(test_size)


# -------------------------------------------------------
# PIPELINES
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_preprocess_pipeline_cached(df_raw: pd.DataFrame, drop_cols_for_model: tuple):
    """Build preprocessing pipeline with robust OneHotEncoder."""
    numeric_features = [c for c in NUMERIC_COLS if c in df_raw.columns]
    numeric_features = [c for c in numeric_features if df_raw[c].notna().any()]

    categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns and c not in drop_cols_for_model]

    numeric_pipe = Pipeline(steps=[ 
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", 
            drop="first", 
            sparse_output=False, 
            dtype=np.float64
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def get_Xy_for_target(df_raw: pd.DataFrame, target_col: str, drop_cols_for_model: tuple):
    """Extract features and target."""
    df = df_raw.copy()
    df = coerce_numeric_like(df, NUMERIC_COLS)

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found in dataset.")

    drop_targets = [TARGET_RISK_TYPE, TARGET_RISK_LEVEL]
    feature_cols = [c for c in df.columns if c not in drop_targets and c not in drop_cols_for_model]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    y = y.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    y = merge_rare_classes(y, min_count=2, other_label="Other")
    
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("missing")
        else:
            X[col] = X[col].fillna(X[col].median())
    
    return X, y


def build_models_fast(fast_mode: bool):
    rf_estimators = 150 if fast_mode else 400
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=rf_estimators, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
