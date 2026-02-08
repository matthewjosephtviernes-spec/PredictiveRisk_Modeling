import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif, chi2
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Constants
TARGET_RISK_TYPE = "Risk_Type"
TARGET_RISK_LEVEL = "Risk_Level"

NUMERIC_COLS = [
    "MP_Count_per_L", "Risk_Score", "Microplastic_Size_mm", 
    "Density", "Latitude", "Longitude",
]

CATEGORICAL_COLS = [
    "Location", "Shape", "Polymer_Type", "pH", "Salinity", 
    "Industrial_Activity", "Population_Density", "Author", "Source",
]

DEFAULT_MODEL_DROP_COLS = ["Location", "Author"]

# Data Loading and Cleaning
def load_data(uploaded_file=None):
    """Load CSV data with encoding fallbacks."""
    if uploaded_file is None:
        path = "Microplastic.csv"
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)
    else:
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

def handle_missing_values(df):
    """Handle missing values."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df

# Categorical Encoding
def encode_categorical_columns(df, categorical_cols):
    """Encode categorical columns."""
    df_encoded = df.copy()
    for col in categorical_cols:
        if col not in df_encoded.columns:
            continue
        if col == TARGET_RISK_TYPE or col == TARGET_RISK_LEVEL:
            label_map = {val: idx for idx, val in enumerate(sorted(df_encoded[col].unique()))}
            df_encoded[col] = df_encoded[col].map(label_map)
        else:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
    return df_encoded

# Outlier Detection and Handling
def detect_and_handle_outliers(df, numeric_cols):
    """Detect and handle outliers using IQR."""
    df_outliers = df.copy()
    for col in numeric_cols:
        if col not in df_outliers.columns:
            continue
        Q1 = df_outliers[col].quantile(0.25)
        Q3 = df_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outliers[col] = np.clip(df_outliers[col], lower_bound, upper_bound)
    return df_outliers

# Feature Scaling
def scale_features(df, numeric_cols):
    """Standard scale numeric features."""
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Skewness Transformation
def transform_skewed(df, numeric_cols):
    """Log transform skewed features."""
    for col in numeric_cols:
        if df[col].skew() > 0.5:
            df[col] = np.log1p(df[col])
    return df

# Feature Selection
def compute_mutual_information(X, y):
    """Compute mutual information."""
    mi_scores = mutual_info_classif(X, y)
    return pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores}).sort_values("MI_Score", ascending=False)

def compute_chi_squared(X, y):
    """Compute chi-squared test."""
    chi2_scores = chi2(X, y)[0]
    return pd.DataFrame({"Feature": X.columns, "Chi2_Score": chi2_scores}).sort_values("Chi2_Score", ascending=False)

def compute_rf_importance(X, y):
    """Compute random forest feature importance."""
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    return pd.DataFrame({"Feature": X.columns, "RF_Importance": rf.feature_importances_}).sort_values("RF_Importance", ascending=False)

# Modeling
def build_models():
    """Define classification models."""
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate models and return performance metrics."""
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        })
    return pd.DataFrame(metrics)

# Cross-validation
def cross_validate(X, y, n_splits=5):
    """Perform K-fold cross-validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        models = build_models()
        result = evaluate_models(X_train, y_train, X_test, y_test, models)
        metrics.append(result)
    return pd.concat(metrics)

# Visualization
def plot_histogram(df, col):
    """Plot histogram of a column."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    st.pyplot()

def plot_boxplot(df, col):
    """Plot boxplot of a column."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    st.pyplot()

# Streamlit App
def main():
    st.title('Data Analysis and Modeling Streamlit App')

    # Sidebar - File Upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write(df.head())

        # Data Preprocessing
        df_clean = handle_missing_values(df)
        df_encoded = encode_categorical_columns(df_clean, CATEGORICAL_COLS)
        df_outliers = detect_and_handle_outliers(df_encoded, NUMERIC_COLS)
        df_scaled = scale_features(df_outliers, NUMERIC_COLS)
        df_transformed = transform_skewed(df_scaled, NUMERIC_COLS)

        # Feature Selection
        X = df_transformed.drop(columns=[TARGET_RISK_TYPE, TARGET_RISK_LEVEL])
        y = df_transformed[TARGET_RISK_TYPE]
        mi_df = compute_mutual_information(X, y)
        chi2_df = compute_chi_squared(X, y)
        rf_df = compute_rf_importance(X, y)

        st.subheader('Feature Selection Results')
        st.write('Mutual Information')
        st.dataframe(mi_df)
        st.write('Chi-Squared')
        st.dataframe(chi2_df)
        st.write('Random Forest Feature Importance')
        st.dataframe(rf_df)

        # Modeling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = build_models()
        metrics = evaluate_models(X_train, y_train, X_test, y_test, models)

        st.subheader('Model Evaluation')
        st.dataframe(metrics)

        # Cross-Validation
        st.subheader('Cross Validation Results')
        cv_metrics = cross_validate(X, y)
        st.dataframe(cv_metrics)

        # Visualization
        st.subheader('Data Distribution')
        plot_histogram(df, 'Risk_Score')
        plot_boxplot(df, 'Risk_Score')

if __name__ == '__main__':
    main()
