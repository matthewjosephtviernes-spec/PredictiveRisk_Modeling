import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF 

# --- Configuration ---
st.set_page_config(
    page_title="Microplastic Pollution Risk System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Theming (Minimalist, blue/green) ---
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(#2e7bcf, #2cb4a0);
        color: white;
    }
    .Widget>label {
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0056b3; /* Darker blue for headers */
    }
    .stButton>button {
        background-color: #2cb4a0; /* Green for buttons */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #208e7a;
    }
    .stFileUploader>label {
        color: #0056b3;
    }
    /* Main content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg { /* This targets the Streamlit main content area directly */
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def load_data(uploaded_file):
    """Loads data from CSV or Excel."""
    if uploaded_file.name.endswith('.csv'):
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df

def preprocess_data(df):
    """Data cleaning and preprocessing, including imputation and encoding."""
    if df is None or df.empty:
        st.error("Cannot preprocess: Input DataFrame is empty.")
        st.session_state['encoded_features'] = []
        return None

    st.subheader("Data Preprocessing Steps (Applied Automatically)")
    st.info(f"Initial dataset shape: {df.shape}")

    df.columns = df.columns.astype(str)
    df_clean = df.copy() 
    
    # --- 1. Smart Missing Value Handling (Imputation) ---
    # Impute Numerical Columns: Use the Median
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True) 

    # Impute Categorical Columns: Use a 'Missing' label
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna('Missing', inplace=True) 

    st.write(f"- Handled missing values using **Imputation** (Median for numeric, 'Missing' for categorical).")

    # --- 2. Drop Duplicates ---
    original_rows_after_impute = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    st.write(f"- Removed {original_rows_after_impute - df_clean.shape[0]} duplicate rows.")
    
    # --- 3. Feature Encoding for Models ---
    features_to_encode = []
    
    # Location Column Encoding
    location_col_name = st.session_state.get('location_col') 
    if location_col_name and location_col_name in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[location_col_name]):
        encoded_col_name = f'{location_col_name}_encoded'
        df_clean[encoded_col_name] = df_clean[location_col_name].astype('category').cat.codes
        st.session_state['location_encoded_col'] = encoded_col_name
        features_to_encode.append(encoded_col_name)
        st.write(f"- Encoded location column '{location_col_name}' into '{encoded_col_name}'.")
    else:
        st.session_state['location_encoded_col'] = location_col_name if location_col_name in df_clean.columns else None

    # Temporal Column Encoding
    temporal_col_name = st.session_state.get('temporal_col')
    if temporal_col_name and temporal_col_name in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[temporal_col_name]):
        encoded_col_name = f'{temporal_col_name}_encoded'
        df_clean[encoded_col_name] = df_clean[temporal_col_name].astype('category').cat.codes
        features_to_encode.append(encoded_col_name)
        st.write(f"- Encoded temporal column '{temporal_col_name}' into '{encoded_col_name}'.")
    
    # Other Non-Numeric Pollution Indicators Encoding
    indicators = st.session_state.get('pollution_indicators', [])
    for col in indicators:
        if col in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[col]) and col not in [location_col_name, temporal_col_name]:
             encoded_col_name = f'{col}_encoded'
             df_clean[encoded_col_name] = df_clean[col].astype('category').cat.codes
             features_to_encode.append(encoded_col_name)
             st.write(f"- Encoded pollution indicator '{col}' into '{encoded_col_name}'.")

    st.session_state['encoded_features'] = features_to_encode
    
    if df_clean.empty:
        st.error("Processed DataFrame is empty.")
        return None
        
    st.success(f"Preprocessing complete! Final usable rows: {df_clean.shape[0]}")
    return df_clean

def train_model(df, target_column, features, model_type):
    """Model training and evaluation (Classification)."""
    # Robust checks against empty data and missing columns
    if df is None or df.empty:
        st.error("Cannot train model: Input DataFrame is empty.")
        return None, None, None
    if target_column not in df.columns or not features:
        st.error("Cannot train model. Please check data, target, and features.")
        return None, None, None

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Cannot train model: Missing feature columns in data: {', '.join(missing_features)}")
        return None, None, None

    X = df[features]
    y = df[target_column]
    
    # --- CRITICAL FIX 1: Check Minimum Data and Classes ---
    if y.empty or X.empty or X.shape[0] < 2:
        st.error("Cannot train model: Dataset or target is empty, or has insufficient rows.")
        return None, None, None
    
    # 1. Target Label Encoding
    label_encoder = None
    target_names = None
    
    # Check for sufficient unique classes 
    try:
        # Check for non-numeric target or too many unique values (classification limit)
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() > 20: 
            if y.nunique() <= 1:
                st.error("Target variable has one or zero unique classes. Cannot train a classifier.")
                return None, None, None
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y.astype(str)) # Ensure string conversion for robust encoding
            target_names = [str(lab) for lab in label_encoder.classes_]
            st.info(f"Target column '{target_column}' converted to numerical categories: {target_names}.")
            y = pd.Series(y_encoded, index=y.index)
        else:
            if y.nunique() <= 1:
                st.error("Target variable has one or zero unique classes. Cannot train a classifier.")
                return None, None, None
            target_names = [str(lab) for lab in sorted(y.unique())]

    except Exception as e:
        st.error(f"Error processing target variable: {e}")
        return None, None, None

    # 2. Split and Model Training
    try:
        y_stratify = y if isinstance(y, pd.Series) else pd.Series(y)
        
        # Ensure that the smallest class has at least 2 samples for stratification (test size 0.3 means 70% train)
        # Even with 0.3 test size, minimum samples per class in the total set should be around 3-4 for safety.
        min_class_count = y_stratify.value_counts().min()
        
        if len(y_stratify.unique()) > 1 and min_class_count >= 2: # At least 2 samples per class to allow a split
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_stratify)
        else:
             st.warning(f"Cannot stratify safely (min class count: {min_class_count}). Falling back to non-stratified split.")
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
             
    except ValueError as e:
        st.warning(f"Train/Test split failed: {e}. Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = None
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(random_state=42) 

    if model:
        # --- CRITICAL FIX 2: Final Check before Fit ---
        if X_train.empty or y_train.size == 0:
            st.error("Training dataset is empty after train/test split. Cannot fit model (This triggered the AttributeError).")
            return None, None, None
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.success(f"Model '{model_type}' trained successfully!")
        st.write(f"**Model Performance:**")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

        st.session_state['model_report'] = {
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        }
        st.session_state['label_encoder'] = label_encoder
        st.session_state['target_names'] = target_names
        
        return model, X_test, y_test
    return None, None, None

# ... (run_kmeans and generate_report functions are not shown but remain the same) ...

# --- Initialize Session State ---
if 'df' not in st.session_state: st.session_state['df'] = None
if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
if 'processed_df_clustered' not in st.session_state: st.session_state['processed_df_clustered'] = None 
if 'model' not in st.session_state: st.session_state['model'] = None
if 'predictions' not in st.session_state: st.session_state['predictions'] = None
if 'test_data' not in st.session_state: st.session_state['test_data'] = None
if 'model_report' not in st.session_state: st.session_state['model_report'] = None
if 'kmeans_model' not in st.session_state: st.session_state['kmeans_model'] = None
if 'kmeans_results' not in st.session_state: st.session_state['kmeans_results'] = None
if 'model_type' not in st.session_state: st.session_state['model_type'] = None
if 'location_col' not in st.session_state: st.session_state['location_col'] = None
if 'temporal_col' not in st.session_state: st.session_state['temporal_col'] = None
if 'pollution_indicators' not in st.session_state: st.session_state['pollution_indicators'] = []
if 'encoded_features' not in st.session_state: st.session_state['encoded_features'] = []
if 'risk_map_plot_buffer' not in st.session_state: st.session_state['risk_map_plot_buffer'] = None
if 'target_col' not in st.session_state: st.session_state['target_col'] = None


# --- Sidebar Navigation (Reruns preprocessing if sidebar selection changes) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Clustering Dashboard", "Reports"]
)
# ... (Sidebar logic remains the same) ...

# --- Main Content Area ---

# ... (Home, Upload Dataset, Data Analysis pages remain the same) ...

elif page == "Prediction Dashboard":
    st.title("Prediction Dashboard (Classification)")

    if st.session_state['processed_df'] is not None:
        df = st.session_state['processed_df']

        st.subheader("Model Configuration")
        target_column_options = [col for col in df.columns if col not in [st.session_state.get('location_col'), st.session_state.get('temporal_col')]]
        
        target_col_current = st.session_state.get('target_col', target_column_options[0] if target_column_options else None)
        if target_col_current not in target_column_options and target_column_options:
            target_col_current = target_column_options[0]

        target_column = st.selectbox("Select Target Variable (e.g., 'Risk_Level')", target_column_options, index=target_column_options.index(target_col_current) if target_col_current in target_column_options else 0, key='target_select')
        st.session_state['target_col'] = target_column 

        # --- CRITICAL FIX 3: Expanded Feature Selection Logic ---
        available_features = [col for col in df.columns if col not in [target_column, st.session_state.get('location_col'), st.session_state.get('temporal_col')]]
        
        # Include all numeric columns in the dataframe (imputed or original)
        numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(df[col])]
        
        # Include all encoded features (these are numeric too, but helps track them)
        encoded_features = [col for col in st.session_state['encoded_features'] if col != target_column] 

        # Filter out duplicates and ensure all are valid columns
        all_valid_features = sorted(list(set(numeric_features + encoded_features)))

        feature_columns = st.multiselect("Select Features for Prediction (Numeric or Encoded)", all_valid_features, default=all_valid_features)
        
        model_type = st.radio("Choose Prediction Model", ["Random Forest", "XGBoost"])
        st.session_state['model_type'] = model_type 
        # ... (rest of the dashboard logic remains the same) ...

        if st.button("Train Model & Generate Predictions"):
            # ... (model training call) ...
            pass
        
        if st.session_state['model'] is not None and st.session_state['predictions'] is not None and st.session_state['model_type'] in ["Random Forest", "XGBoost"]:
            # ... (display results) ...
            pass
        else:
             st.info("Train the model first to see the scorecard and results.")

    else:
        st.warning("Please upload and preprocess a dataset on the 'Upload Dataset' page first.")

# ... (Clustering Dashboard and Reports pages remain the same) ...
