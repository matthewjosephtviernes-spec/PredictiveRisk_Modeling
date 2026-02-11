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
    from imblearn.oversampling import SMOTE
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
    "Microplastic_Size_mm",  # Replaced 'Microplastic_Size_mm_midpoint' with available column
    "Density",  # Replaced 'Density_midpoint' with available column
]

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

def convert_to_numeric(df, columns):
    """Convert specified columns to numeric, coercing errors to NaN."""
    for col in columns:
        if col in df.columns:  # Only convert if column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.warning(f"Column '{col}' not found in dataset.")
    return df

def handle_outliers_iqr(df, numerical_cols):
    """Handle outliers using the IQR method (capping)."""
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap the outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            st.warning(f"Column '{col}' not found for outlier handling.")
    return df

def encode_categorical(df, categorical_cols):
    """Apply One-Hot Encoding to categorical columns."""
    # Check which columns are present in the DataFrame
    existing_cols = [col for col in categorical_cols if col in df.columns]
    
    if len(existing_cols) != len(categorical_cols):
        missing_cols = list(set(categorical_cols) - set(existing_cols))
        st.warning(f"Missing columns for encoding: {', '.join(missing_cols)}")

    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    # Label Encoding for the target columns (Risk_Type, Risk_Level)
    label_encoder = LabelEncoder()
    if 'Risk_Type' in df_encoded.columns:
        df_encoded['Risk_Type'] = label_encoder.fit_transform(df_encoded['Risk_Type'])
    if 'Risk_Level' in df_encoded.columns:
        df_encoded['Risk_Level'] = label_encoder.fit_transform(df_encoded['Risk_Level'])

    return df_encoded

def scale_features(df, numerical_cols):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def transform_skewed_columns(df, numerical_cols, threshold=0.5):
    """Transform skewed columns by applying log transformation."""
    skewness_before = df[numerical_cols].apply(lambda x: x.skew(), axis=0)
    
    skewed_cols = skewness_before[skewness_before > threshold].index.tolist()
    
    # Apply log transformation to the skewed columns
    for col in skewed_cols:
        df[col] = np.log1p(df[col])  # Apply log1p to handle skewed distributions
    
    skewness_after = df[skewed_cols].apply(lambda x: x.skew(), axis=0)
    
    return df, skewness_before, skewness_after

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
    
    # Check if the dataset is already in session state
    if 'df' in st.session_state:
        st.write("Dataset already uploaded!")
        st.dataframe(st.session_state.df.head())
    
    if uploaded_file is not None:
        try:
            df_raw = load_data(uploaded_file)
            st.write("Dataset uploaded successfully!")
            st.dataframe(df_raw.head())

            # Store the dataframe in session state so it persists across pages
            st.session_state.df = df_raw

            # Convert columns to numeric (if they are not already)
            df_raw = convert_to_numeric(df_raw, ['MP_Count_per_L', 'Risk_Score'])

            # Store the content for Home Page in session state
            st.session_state.home_content = df_raw

            # EDA Visualizations (after dataset upload)
            st.subheader("Exploratory Data Analysis (EDA)")
            
            if "Risk_Score" in df_raw.columns and "MP_Count_per_L" in df_raw.columns:
                # Create a toggle for the Distribution of Risk Score
                with st.expander('Distribution of Risk Score'):
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df_raw, x='Risk_Score', kde=True, bins=30)
                    plt.title('Distribution of Risk Score')
                    plt.xlabel('Risk Score')
                    plt.ylabel('Frequency')
                    st.pyplot(plt)

                # Create a toggle for the Box Plot of Risk Score
                with st.expander('Box Plot of Risk Score'):
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df_raw, y='Risk_Score')
                    plt.title('Box Plot of Risk Score')
                    plt.ylabel('Risk Score')
                    st.pyplot(plt)

                # Create a toggle for the scatter plot of Risk_Score and MP_Count_per_L
                with st.expander('Relationship between MP Count per L and Risk Score'):
                    plt.figure(figsize=(10, 6))
                    
                    # Apply log transformation to reduce scale differences if necessary
                    df_raw['Log_MP_Count_per_L'] = np.log1p(df_raw['MP_Count_per_L'])
                    df_raw['Log_Risk_Score'] = np.log1p(df_raw['Risk_Score'])
                    
                    # Scatter plot with transparency (alpha) for better visibility
                    sns.scatterplot(data=df_raw, x='Log_MP_Count_per_L', y='Log_Risk_Score', alpha=0.6)
                    
                    # Customize plot appearance
                    plt.title('Relationship between MP Count per L and Risk Score')
                    plt.xlabel('Log of MP Count per L')
                    plt.ylabel('Log of Risk Score')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    st.pyplot(plt)

                # Create a toggle for the Box Plot or Violin Plot of Risk_Score by Risk_Level
                with st.expander('Risk Score Distribution by Risk Level'):
                    plot_type = st.radio('Select Plot Type', ['Box Plot', 'Violin Plot'], index=0)

                    if plot_type == 'Box Plot':
                        plt.figure(figsize=(12, 8))
                        sns.boxplot(data=df_raw, x='Risk_Level', y='Risk_Score')
                        plt.title('Risk Score Distribution by Risk Level')
                        plt.xlabel('Risk Level')
                        plt.ylabel('Risk Score')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(plt)

                    elif plot_type == 'Violin Plot':
                        plt.figure(figsize=(12, 8))
                        sns.violinplot(data=df_raw, x='Risk_Level', y='Risk_Score')
                        plt.title('Risk Score Distribution by Risk Level')
                        plt.xlabel('Risk Level')
                        plt.ylabel('Risk Score')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(plt)

                # Non-clickable Indicator to proceed to Preprocessing
                st.subheader("You can proceed to the Preprocessing section.")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.write("No file uploaded yet. Please upload a CSV file to proceed.")

# -------------------------------------------------------
# PREPROCESSING PAGE
# -------------------------------------------------------
def preprocessing_page():
    st.title("Preprocessing")

    # Retrieve the dataset from session state
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset first on the Home page.")
        return

    df_raw = st.session_state.df

    # Encoding Categorical Variables Tab
    st.subheader("Encoding Categorical Variables")
    df_encoded = encode_categorical(df_raw, CATEGORICAL_COLS)
    st.write("First few rows after One-Hot Encoding:")
    st.dataframe(df_encoded.head())

    # Handle outliers in the numerical columns (only columns that exist)
    numerical_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm', 'Density']
    
    # Convert columns to numeric (if they are not already)
    df_encoded = convert_to_numeric(df_encoded, numerical_cols)

    # Handle outliers using IQR method for existing columns
    df_cleaned = handle_outliers_iqr(df_encoded, numerical_cols)

    # Transform skewed columns (log transformation)
    df_cleaned, skewness_before, skewness_after = transform_skewed_columns(df_cleaned, numerical_cols)

    # Display skewness before and after transformation
    st.write("Skewness of numerical columns before and after log transformation:")
    skewness_df = pd.DataFrame({
        'Column': skewness_before.index,
        'Skewness Before': skewness_before.values,
        'Skewness After': skewness_after.values
    })
    st.dataframe(skewness_df)

    # Scale numerical features
    df_scaled, scaler = scale_features(df_cleaned, numerical_cols)

    # Show scaled features
    st.subheader("Scaled Features (after applying StandardScaler)")
    st.dataframe(df_scaled.head())

    # Display descriptive statistics after outlier handling
    st.write("Descriptive statistics after handling outliers and scaling:")
    st.dataframe(df_scaled[numerical_cols].describe())

# -------------------------------------------------------
# MAIN APP
# -------------------------------------------------------
def main():
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
        
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("🏠 Home", "🧼 Preprocessing", "🧠 Feature Selection & Relevance", "⚙️ Modeling", "📊 Cross Validation"))

    if page == "🏠 Home":
        home_page()

    elif page == "🧼 Preprocessing":
        preprocessing_page()

    elif page == "🧠 Feature Selection & Relevance":
        st.title("Feature Selection & Relevance")
        st.write("Feature selection content goes here...")

    elif page == "⚙️ Modeling":
        st.title("Modeling")
        st.write("Modeling content goes here...")

    elif page == "📊 Cross Validation":
        st.title("Cross Validation")
        st.write("Cross-validation content goes here...")

if __name__ == "__main__":
    main()
