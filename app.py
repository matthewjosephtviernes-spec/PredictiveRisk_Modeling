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
    
    # Add file upload section
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df_raw = load_data(uploaded_file)
            st.write("Dataset uploaded successfully!")
            st.dataframe(df_raw.head())
            
            # EDA Visualizations (after dataset upload)
            st.subheader("Exploratory Data Analysis (EDA)")
            
            if "Risk_Score" in df_raw.columns:
                # Create a histogram of the Risk_Score column
                st.subheader('Distribution of Risk Score')
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df_raw, x='Risk_Score', kde=True, bins=30)
                plt.title('Distribution of Risk Score')
                plt.xlabel('Risk Score')
                plt.ylabel('Frequency')
                st.pyplot(plt)

                # Create a box plot of the Risk_Score column
                st.subheader('Box Plot of Risk Score')
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_raw, y='Risk_Score')
                plt.title('Box Plot of Risk Score')
                plt.ylabel('Risk Score')
                st.pyplot(plt)
                
            else:
                st.warning("No 'Risk_Score' column found in the dataset.")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.write("No file uploaded yet. Please upload a CSV file to proceed.")

# -------------------------------------------------------
# MAIN APP
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
