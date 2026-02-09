import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import EmptyDataError, ParserError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Optional: Import libraries for feature selection and machine learning
from sklearn.feature_selection import SelectKBest

# Constants
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

# Load Data
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


# Feature Selection Methods
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


# Main Page with Feature Selection
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

    # Train a Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy:.2f}")

    # Visualize feature importance
    st.subheader("Feature Importance Visualization")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=rf_df)
    st.pyplot()

# Run app
def main():
    st.title("Microplastic Risk Prediction")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Preprocessing", "Feature Selection & Relevance"))

    if page == "Preprocessing":
        st.title("Preprocessing Page")
        data = pd.read_csv("your_dataset.csv")  # Replace with your dataset
        st.dataframe(data.head())

    elif page == "Feature Selection & Relevance":
        feature_selection_page(data)

if __name__ == "__main__":
    main()
