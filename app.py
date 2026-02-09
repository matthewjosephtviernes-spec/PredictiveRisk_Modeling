import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import EmptyDataError, ParserError

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
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

# Optional: SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_OK = True
except Exception:
    IMBLEARN_OK = False


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


def rank_features_multi_method(X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> dict:
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
# MAIN FEATURE SELECTION PAGE
# -------------------------------------------------------
def feature_selection_page(df_raw):
    st.title("Feature Selection & Relevance")

    # Clarifying Target Variables (Risk Type and Risk Level)
    target_variables = st.selectbox("Select the target variable", ["Risk Type", "Risk Level"])

    target = target_variables
    features = df_raw.drop(columns=[target])

    st.write(f"Target Variable: {target}")
    st.write("Selecting appropriate feature selection methods...")

    # Apply Label Encoding for categorical variables
    le = LabelEncoder()
    for col in features.select_dtypes(include=['object']).columns:
        features[col] = le.fit_transform(features[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # Apply feature selection methods
    st.subheader("Top 20 Features Based on Mutual Information")
    mi_scores = mutual_info_classif(features, df_raw[target])
    mi_df = pd.DataFrame({
        'Feature': features.columns,
        'MI Score': mi_scores
    })
    mi_df = mi_df.sort_values(by='MI Score', ascending=False).head(20)
    st.write(mi_df)

    st.subheader("Top 20 Features Based on Chi-Squared Test")
    chi2_scores, p_values = chi2(features, df_raw[target])
    chi2_df = pd.DataFrame({
        'Feature': features.columns,
        'Chi2 Score': chi2_scores,
        'P-value': p_values
    })
    chi2_df = chi2_df.sort_values(by='Chi2 Score', ascending=False).head(20)
    st.write(chi2_df)

    st.subheader("Feature Importance Using Random Forest")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(features, df_raw[target])
    feature_importances = rf.feature_importances_
    rf_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': feature_importances
    })
    rf_df = rf_df.sort_values(by='Importance', ascending=False).head(20)
    st.write(rf_df)

    # Create DataFrame with Top Features based on Mutual Information
    top_mi_features = mi_df['Feature'].tolist()
    selected_features_mi = features[top_mi_features]
    st.write(f"New DataFrame with Top Features Based on Mutual Information:")
    st.dataframe(selected_features_mi)

    # Split into Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(selected_features_mi, df_raw[target], test_size=0.2, random_state=42)
    st.write("Training and Testing data split completed.")

    # Display shapes of the train and test datasets
    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Test data shape: {X_test.shape}")

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
def home_page():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        """
        This app demonstrates the analysis and modeling workflow for predicting **Risk_Type**
        and **Risk_Level** using microplastic and environmental features.
        """
    )


# -------------------------------------------------------
# MODELING PAGE
# -------------------------------------------------------
def modeling_page(df_raw):
    st.title("Classification Modeling")

    st.write("Modeling details will go here...")
    # Add logic for classification modeling like Logistic Regression, Random Forest, Gradient Boosting
    pass


# -------------------------------------------------------
# CROSS VALIDATION PAGE
# -------------------------------------------------------
def cross_validation_page(df_raw):
    st.title("Cross Validation (K-Fold)")

    st.write("Cross-validation details will go here...")
    # Add logic for cross-validation, etc.
    pass


# -------------------------------------------------------
# APP MAIN
# -------------------------------------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("🏠 Home", "🧼 Preprocessing", "🧠 Feature Selection & Relevance", "⚙️ Modeling", "📊 Cross Validation"))

    if page == "🏠 Home":
        home_page()

    elif page == "🧼 Preprocessing":
        st.title("Preprocessing Page")
        data = pd.read_csv("your_dataset.csv")  # Replace with your dataset
        st.dataframe(data.head())

    elif page == "🧠 Feature Selection & Relevance":
        feature_selection_page(data)

    elif page == "⚙️ Modeling":
        modeling_page(data)

    elif page == "📊 Cross Validation":
        cross_validation_page(data)


if __name__ == "__main__":
    main()
