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
    original_rows = df_clean.shape[0]
    
    # Impute Numerical Columns: Use the Median
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        # Fill NaN with the median value of that column
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True) 

    # Impute Categorical Columns: Use a 'Missing' label
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        # Fill NaN with a specific label 'Missing'
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

# ... (train_model and run_kmeans functions remain the same as the previous correct version) ...
# Note: They are already robust against empty data/single-class targets.

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
    
    # --- FIX for AttributeError: Check if target Series is empty ---
    if y.empty:
        st.error("Cannot train model: Target variable Series is empty after slicing.")
        return None, None, None
    
    # 1. Target Label Encoding
    label_encoder = None
    target_names = None

    # Check for sufficient unique classes (where the original error occurred)
    try:
        if y.nunique() <= 1:
            st.error("Target variable has one or zero unique classes. Cannot train a classifier.")
            return None, None, None
    except Exception as e:
        st.error(f"Error checking unique classes in target variable: {e}. Check data type.")
        return None, None, None

    if not pd.api.types.is_numeric_dtype(y) or y.nunique() > 20: 
        try:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y.astype(str)) # Ensure string conversion for robust encoding
            target_names = [str(lab) for lab in label_encoder.classes_]
            st.info(f"Target column '{target_column}' converted to numerical categories: {target_names}.")
            y = pd.Series(y_encoded, index=y.index)
        except Exception as e:
            st.error(f"Could not convert target column '{target_column}' to numerical categories. Error: {e}")
            return None, None, None
    else:
        target_names = [str(lab) for lab in sorted(y.unique())]

    # 2. Split and Model Training
    try:
        # Use a Series for stratification check
        y_stratify = y if isinstance(y, pd.Series) else pd.Series(y)
        
        if len(y_stratify.unique()) > 1:
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_stratify)
        else:
             raise ValueError("Cannot stratify with one class.")
    except ValueError as e:
        st.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = None
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(random_state=42) 

    if model:
        if X_train.empty or y_train.size == 0:
            st.error("Training dataset is empty after train/test split. Cannot fit model.")
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

def run_kmeans(df, features, n_clusters):
    """Run K-Means clustering and evaluate."""
    if df is None or df.empty or not features:
        st.error("Cannot run K-Means. Please check data and features.")
        return None, None, None, None

    # Filter out rows with NaNs in the selected features before clustering
    X = df[features].dropna().copy()
    
    if X.empty:
        st.error("Cannot run K-Means: Selected features contain only missing values or are empty after cleaning.")
        return None, None, None, None
        
    # Scale the data for K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        # Use the index of the clean, scaled data for alignment
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        X['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Merge clusters back into the original dataframe structure
        df_clustered = df.copy()
        df_clustered = df_clustered.merge(X[['Cluster']], left_index=True, right_index=True, how='left')
        df_clustered['Cluster'] = df_clustered['Cluster'].fillna(-1).astype(int) # -1 for unclustered (NaN) rows

    except Exception as e:
        st.error(f"K-Means failed: {e}. Check if features have enough variability.")
        return None, None, None, None

    wcss = kmeans.inertia_
    silhouette = None
    if n_clusters > 1 and len(X) > n_clusters:
        try:
            silhouette = silhouette_score(X_scaled, X['Cluster'])
        except Exception:
            silhouette = None

    st.success(f"K-Means Clustering complete with {n_clusters} clusters!")
    st.write(f"**Clustering Metrics:**")
    st.write(f"WCSS (Inertia): {wcss:.2f}")
    if silhouette is not None:
        st.write(f"Silhouette Score: {silhouette:.2f} (Higher is better)")
    else:
        st.write("Silhouette Score: N/A (requires >1 cluster and sufficient samples)")

    return df_clustered, kmeans, wcss, silhouette


def generate_report(df, predictions, model_results, plot_buffer, kmeans_results=None):
    """Generates a PDF report (Classification and Clustering)."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Microplastic Pollution Risk Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"This report summarizes the analysis and results based on the provided dataset.")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Analysis Overview", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, f"Dataset processed: {st.session_state.get('uploaded_filename', 'N/A')} with {df.shape[0]} rows and {df.shape[1]} columns.")
    pdf.multi_cell(0, 7, f"Report Type: {'Classification' if model_results else 'Clustering'}")
    pdf.ln(5)

    # --- Classification Model Performance ---
    if model_results and st.session_state.get('model_type') != 'K-Means':
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"2. Predictive Model Performance ({model_results['model_type']})", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 7, f"Accuracy: {model_results['accuracy']:.2f}", 0, 1)
        pdf.cell(0, 7, f"Precision (Weighted): {model_results['precision']:.2f}", 0, 1)
        pdf.cell(0, 7, f"Recall (Weighted): {model_results['recall']:.2f}", 0, 1)
        pdf.cell(0, 7, f"F1-Score (Weighted): {model_results['f1_score']:.2f}", 0, 1)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Classification Report:", 0, 1)
        pdf.set_font("Arial", "", 10)
        # Handle the classification report data structure
        report_data = model_results['classification_report']
        for class_name, metrics in report_data.items():
            if isinstance(metrics, dict):
                pdf.cell(0, 5, f"  {class_name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}, Support={metrics['support']}", 0, 1)
        pdf.ln(5)

    # --- K-Means Clustering Results ---
    if kmeans_results and st.session_state.get('model_type') == 'K-Means':
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"2. K-Means Clustering Results (k={kmeans_results['n_clusters']})", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 7, f"WCSS (Inertia): {kmeans_results['wcss']:.2f}", 0, 1)
        if kmeans_results['silhouette'] is not None:
             pdf.cell(0, 7, f"Silhouette Score: {kmeans_results['silhouette']:.2f}", 0, 1)
        pdf.ln(5)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Cluster Sizes:", 0, 1)
        pdf.set_font("Arial", "", 10)
        # Ensure 'Cluster' column exists before value_counts
        if 'Cluster' in df.columns:
            cluster_counts = df['Cluster'].value_counts().sort_index()
            for i, count in cluster_counts.items():
                if i != -1: # Skip the -1 (NaN) cluster
                    pdf.cell(0, 5, f"  Cluster {i}: {count} samples", 0, 1)
        pdf.ln(5)


    if plot_buffer:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. Visualizations and Map", 0, 1)
        pdf.ln(2)
        try:
            plot_io = BytesIO(plot_buffer)
            # FPDF image placement needs care with y-coordinate
            pdf.image(plot_io, x=10, y=pdf.get_y(), w=190)
            pdf.ln(100) # Move down significantly to account for image height
        except Exception as e:
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, f"Warning: Could not embed image in PDF. Error: {e}")
            pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4. Summary and Mitigation Strategies", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, "This section provides a brief interpretation of the results, detailing high-risk zones (from classification) or characteristic groups (from clustering). Suggested mitigation strategies include enhanced waste management, public awareness campaigns, and stricter industrial regulations.")
    pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')


# --- Initialize Session State ---
if 'df' not in st.session_state: st.session_state['df'] = None
if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
if 'processed_df_clustered' not in st.session_state: st.session_state['processed_df_clustered'] = None # New state for clustered data
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


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Clustering Dashboard", "Reports"]
)

st.sidebar.subheader("Input Variables (Global)")
if st.session_state['processed_df'] is not None:
    all_columns = st.session_state['processed_df'].columns.tolist()
    
    location_col_current = st.session_state.get('location_col') if st.session_state.get('location_col') in all_columns else 'None'
    temporal_col_current = st.session_state.get('temporal_col') if st.session_state.get('temporal_col') in all_columns else 'None'

    location_index = ['None'] + all_columns
    location_default_index = location_index.index(location_col_current) if location_col_current in location_index else 0

    temporal_index = ['None'] + all_columns
    temporal_default_index = temporal_index.index(temporal_col_current) if temporal_col_current in temporal_index else 0

    location_col = st.sidebar.selectbox("Select Location Column", location_index, index=location_default_index, key='sidebar_location', help="Used for geographic plots.")
    pollution_indicators = st.sidebar.multiselect("Select Pollution Indicators", all_columns, default=st.session_state['pollution_indicators'], key='sidebar_indicators', help="Numeric/Categorical columns representing pollutants/environmental factors.")
    temporal_col = st.sidebar.selectbox("Select Temporal Column", temporal_index, index=temporal_default_index, key='sidebar_temporal', help="Used for time-series analysis.")

    # Important: Update state immediately so preprocessing can use it
    if st.session_state['location_col'] != (location_col if location_col != 'None' else None) or \
       st.session_state['temporal_col'] != (temporal_col if temporal_col != 'None' else None) or \
       st.session_state['pollution_indicators'] != pollution_indicators:
           
        st.session_state['location_col'] = location_col if location_col != 'None' else None
        st.session_state['pollution_indicators'] = pollution_indicators
        st.session_state['temporal_col'] = temporal_col if temporal_col != 'None' else None
        
        # Rerun preprocessing after sidebar changes to update encoded features list
        if 'df' in st.session_state and st.session_state['df'] is not None:
             st.session_state['processed_df'] = preprocess_data(st.session_state['df'].copy())

else:
    st.sidebar.info("Upload a dataset first to select variables.")


# --- Main Content Area ---

if page == "Home":
    st.title("Welcome to the Microplastic Pollution Risk Assessment System")
    # Added warning for deprecation
    st.warning("The use_column_width parameter has been deprecated and will be removed in a future release. Please utilize the use_container_width parameter instead.")
    st.image("https://via.placeholder.com/700x300.png?text=Environmental+Sustainability", use_container_width=True) 
    st.markdown("""
        <p style='font-size: 1.1em;'>
        This platform leverages advanced predictive analytics to assess and visualize the risk levels of microplastic pollution across various environments.
        Utilizing Streamlit for an interactive user experience and powerful data mining algorithms,
        we provide insights into pollution trends, potential high-risk zones, and actionable mitigation strategies.
        </p>
        <p style='font-size: 1.1em;'>
        Navigate through the sections to upload your data, analyze it, generate predictions, and download comprehensive reports.
        </p>
    """, unsafe_allow_html=True)

    st.subheader("Key Features:")
    st.markdown("""
    - **Data Upload & Preprocessing:** Seamlessly upload and clean environmental datasets.
    - **Descriptive Analytics:** Understand your data with interactive charts and statistics.
    - **Prediction Dashboard (Classification):** Use **Random Forest** or **XGBoost** to classify microplastic pollution risk (Low, Medium, High).
    - **Clustering Dashboard:** Use **K-Means** to discover inherent groupings in the data.
    - **Interactive Dashboards:** Visualize pollution intensity with heatmaps and time-series graphs.
    - **Comprehensive Reporting:** Download detailed reports in PDF or Excel.
    """)

elif page == "Upload Dataset":
    st.title("Upload Your Environmental Dataset")
    st.markdown("Please upload your environmental data file in **CSV** or **Excel** format (.csv, .xls, .xlsx).")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        st.session_state['uploaded_filename'] = uploaded_file.name
        with st.spinner("Loading and preprocessing data..."):
            st.session_state['df'] = load_data(uploaded_file)
            if st.session_state['df'] is not None:
                st.write("Original Data Preview:")
                st.dataframe(st.session_state['df'].head())
                st.session_state['processed_df'] = preprocess_data(st.session_state['df'].copy())

                if st.session_state['processed_df'] is not None:
                    st.success("Dataset loaded and preprocessed successfully!")
                    st.write("Processed Data Preview:")
                    st.dataframe(st.session_state['processed_df'].head())
                    st.write(f"Processed dataset shape: {st.session_state['processed_df'].shape}")
                else:
                    st.error("Data preprocessing failed. Check file content.")
            else:
                st.error("Failed to load data.")
    else:
        st.info("Awaiting file upload.")

elif page == "Data Analysis":
    st.title("Data Analysis & Exploration")

    if st.session_state['processed_df'] is not None:
        df = st.session_state['processed_df']

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

        st.subheader("Data Distribution")
        selected_column_dist = st.selectbox("Select a column to view its distribution:", df.columns)
        if selected_column_dist:
            if pd.api.types.is_numeric_dtype(df[selected_column_dist]):
                fig = px.histogram(df, x=selected_column_dist, title=f'Distribution of {selected_column_dist}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                counts = df[selected_column_dist].value_counts().reset_index()
                counts.columns = [selected_column_dist, 'count']
                fig = px.bar(counts, x=selected_column_dist, y='count', title=f'Count of {selected_column_dist}')
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig_corr = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto",
                                 color_continuous_scale=px.colors.sequential.Plasma,
                                 title="Correlation Matrix of Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns found for correlation analysis.")
            
        st.subheader("Time-series Trends (if temporal data is available)")
        if st.session_state['temporal_col'] and st.session_state['temporal_col'] in df.columns:
            try:
                df_time = df.copy() 
                df_time[st.session_state['temporal_col']] = pd.to_datetime(df_time[st.session_state['temporal_col']], errors='coerce')
                df_time.dropna(subset=[st.session_state['temporal_col']], inplace=True) 
                
                if st.session_state['pollution_indicators']:
                    numeric_indicators = [col for col in st.session_state['pollution_indicators'] if pd.api.types.is_numeric_dtype(df_time[col])]
                    if numeric_indicators:
                        trend_df = df_time.groupby(df_time[st.session_state['temporal_col']].dt.to_period('D'))[numeric_indicators].mean().reset_index()
                        trend_df[st.session_state['temporal_col']] = trend_df[st.session_state['temporal_col']].dt.to_timestamp()
                        fig_time = px.line(trend_df, x=st.session_state['temporal_col'], y=numeric_indicators,
                                        title="Pollution Indicator Trends Over Time")
                        st.plotly_chart(fig_time, use_container_width=True)
                    else:
                        st.info("Selected pollution indicators are not numeric. Select numeric columns in the sidebar to view time trends.")
                else:
                    st.info("Select pollution indicators in the sidebar to view time trends.")
            except Exception as e:
                st.warning(f"Could not plot time series. Ensure '{st.session_state['temporal_col']}' is a valid date column. Error: {e}")
        else:
            st.info("Select a temporal column in the sidebar to view time-series trends.")

    else:
        st.warning("Please upload a dataset on the 'Upload Dataset' page first.")


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

        # --- Feature Selection Logic ---
        available_features = [col for col in df.columns if col not in [target_column, st.session_state.get('location_col'), st.session_state.get('temporal_col')]]
        numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(df[col])]
        encoded_features = [col for col in st.session_state['encoded_features'] if col != target_column] 

        all_valid_features = sorted(list(set(numeric_features + encoded_features)))

        feature_columns = st.multiselect("Select Features for Prediction (Numeric or Encoded)", all_valid_features, default=all_valid_features)
        
        model_type = st.radio("Choose Prediction Model", ["Random Forest", "XGBoost"])
        st.session_state['model_type'] = model_type 

        if st.button("Train Model & Generate Predictions"):
            if target_column and feature_columns:
                with st.spinner(f"Training {model_type} model..."):
                    # The function train_model is robustified to handle empty/single-class targets
                    model, X_test, y_test = train_model(df.copy(), target_column, feature_columns, model_type)
                    if model is not None:
                        st.session_state['model'] = model
                        st.session_state['test_data'] = X_test.copy()
                        st.session_state['test_labels'] = y_test

                        raw_predictions = model.predict(X_test)

                        label_encoder = st.session_state.get('label_encoder')
                        if label_encoder is not None:
                             st.session_state['predictions'] = label_encoder.inverse_transform(raw_predictions)
                             true_labels_decoded = label_encoder.inverse_transform(y_test)
                        else:
                             st.session_state['predictions'] = raw_predictions
                             true_labels_decoded = y_test

                        st.session_state['test_labels_decoded'] = true_labels_decoded
                        
                        st.success("Model trained and predictions generated!")
                    
            else:
                st.warning("Please select a target variable and at least one feature.")

        if st.session_state['model'] is not None and st.session_state['predictions'] is not None and st.session_state['model_type'] in ["Random Forest", "XGBoost"]:
            st.subheader("Prediction Results")

            prediction_df = st.session_state['test_data'].copy()
            prediction_df['True_Risk'] = st.session_state.get('test_labels_decoded', st.session_state['test_labels']) 
            prediction_df['Predicted_Risk'] = st.session_state['predictions']

            st.write("Sample Predictions:")
            st.dataframe(prediction_df.head())

            st.subheader("Risk Map (Geographic Heatmap)")
            if st.session_state.get('location_col') and 'latitude' in df.columns and 'longitude' in df.columns: 
                test_indices = st.session_state['test_data'].index
                plot_df_full = df.loc[test_indices].copy()
                plot_df_full['Predicted_Risk_Level'] = st.session_state['predictions']

                risk_labels = st.session_state.get('target_names', sorted(plot_df_full['Predicted_Risk_Level'].unique()))
                color_list = px.colors.qualitative.Plotly
                risk_color_map = {label: color_list[i % len(color_list)] for i, label in enumerate(risk_labels)}
                if 'Low' in risk_labels: risk_color_map['Low'] = 'green'
                if 'Medium' in risk_labels: risk_color_map['Medium'] = 'orange'
                if 'High' in risk_labels: risk_color_map['High'] = 'red'

                fig_map = px.scatter_mapbox(plot_df_full, lat="latitude", lon="longitude", color="Predicted_Risk_Level",
                                            size_max=15, zoom=3, hover_name=st.session_state['location_col'],
                                            color_discrete_map=risk_color_map, title="Microplastic Pollution Risk Map")
                fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)

                buf = BytesIO()
                fig_map.write_image(buf, format="png", width=800, height=450, scale=2)
                st.session_state['risk_map_plot_buffer'] = buf.getvalue()
            else:
                st.warning("To view the risk map, ensure your dataset has 'latitude' and 'longitude' columns and a location column is selected in the sidebar.")
                st.session_state['risk_map_plot_buffer'] = None

            st.subheader("Model Accuracy Scorecard")
            if st.session_state['model_report']:
                st.markdown(f"""
                - **Model Type:** {st.session_state['model_report']['model_type']}
                - **Overall Accuracy:** <span style='color: #007bff; font-weight: bold;'>{st.session_state['model_report']['accuracy']:.2f}</span>
                - **Weighted F1-Score:** {st.session_state['model_report']['f1_score']:.2f}
                """, unsafe_allow_html=True)
                st.write("Classification Report:")
                report_df = pd.DataFrame(st.session_state['model_report']['classification_report']).transpose().round(2)
                st.dataframe(report_df.drop(columns=['support'], errors='ignore'))
            
        else:
             # This handles the case where DF is loaded but model hasn't been run or failed.
             st.info("Train the model first to see the scorecard and results.")

    else:
        # This handles the case where DF hasn't been uploaded yet.
        st.warning("Please upload and preprocess a dataset on the 'Upload Dataset' page first.")


# --- K-Means Clustering Dashboard ---
elif page == "Clustering Dashboard":
    st.title("Clustering Dashboard (K-Means)")

    if st.session_state['processed_df'] is not None:
        df = st.session_state['processed_df']

        st.subheader("K-Means Configuration")
        
        # Feature selection logic re-used from Prediction Dashboard
        available_features = [col for col in df.columns if col not in [st.session_state.get('location_col'), st.session_state.get('temporal_col')]]
        numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(df[col])]
        encoded_features = st.session_state['encoded_features']
        all_valid_features = sorted(list(set(numeric_features + encoded_features)))

        cluster_features = st.multiselect("Select Features for Clustering (Numeric or Encoded)", all_valid_features, default=all_valid_features, key='cluster_features_select')
        
        n_clusters = st.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=3, step=1)
        
        if st.button("Run K-Means Clustering"):
            if cluster_features:
                with st.spinner(f"Running K-Means with k={n_clusters}..."):
                    df_clustered, kmeans_model, wcss, silhouette = run_kmeans(df.copy(), cluster_features, n_clusters)
                    
                    if kmeans_model:
                        st.session_state['kmeans_model'] = kmeans_model
                        st.session_state['processed_df_clustered'] = df_clustered
                        st.session_state['model_type'] = 'K-Means'
                        st.session_state['kmeans_results'] = {'n_clusters': n_clusters, 'wcss': wcss, 'silhouette': silhouette}
                    
            else:
                st.warning("Please select at least one feature for clustering.")

        if st.session_state.get('kmeans_model') is not None and st.session_state.get('processed_df_clustered') is not None and st.session_state.get('model_type') == 'K-Means':
            df_clustered = st.session_state['processed_df_clustered']
            st.subheader(f"Clustering Results (k={st.session_state['kmeans_results']['n_clusters']})")
            
            cluster_info_df = df_clustered['Cluster'].value_counts().sort_index().to_frame(name='Count')
            if -1 in cluster_info_df.index:
                 cluster_info_df.loc[-1, 'Note'] = 'Skipped (Missing Data)'
            st.write("Cluster Counts:")
            st.dataframe(cluster_info_df)

            st.write("Cluster Characteristics (Mean of Features):")
            st.dataframe(df_clustered[df_clustered['Cluster'] != -1].groupby('Cluster')[cluster_features].mean())

            # Cluster Map (Geographic Visualization)
            st.subheader("Cluster Map (Geographic Visualization)")
            if st.session_state.get('location_col') and 'latitude' in df.columns and 'longitude' in df.columns: 
                plot_df = df_clustered[df_clustered['Cluster'] != -1]
                
                fig_map = px.scatter_mapbox(plot_df,
                                            lat="latitude",
                                            lon="longitude",
                                            color="Cluster",
                                            color_continuous_scale=px.colors.qualitative.Bold,
                                            size_max=15, zoom=3,
                                            hover_name=st.session_state['location_col'],
                                            title=f"Microplastic Pollution Clusters (k={st.session_state['kmeans_results']['n_clusters']})")
                fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)

                buf = BytesIO()
                fig_map.write_image(buf, format="png", width=800, height=450, scale=2)
                st.session_state['risk_map_plot_buffer'] = buf.getvalue()
            else:
                st.warning("To view the cluster map, ensure your dataset has 'latitude' and 'longitude' columns and a location column is selected in the sidebar.")
                st.session_state['risk_map_plot_buffer'] = None
                
    else:
        st.warning("Please upload and preprocess a dataset on the 'Upload Dataset' page first.")

elif page == "Reports":
    st.title("Generate & Download Reports")

    # Determine which data to use for the report based on the last action
    is_classification = st.session_state.get('model_type') in ["Random Forest", "XGBoost"] and st.session_state.get('model') is not None
    is_clustering = st.session_state.get('model_type') == 'K-Means' and st.session_state.get('kmeans_model') is not None

    if is_classification or is_clustering:
        
        report_df_data = st.session_state['processed_df_clustered'] if is_clustering else st.session_state['processed_df']
        model_results = st.session_state['model_report'] if is_classification else None
        kmeans_results = st.session_state['kmeans_results'] if is_clustering else None
        predictions = st.session_state['predictions'] if is_classification else None
        report_type = st.session_state['model_type'].lower() if st.session_state['model_type'] else 'analysis'


        st.write(f"Generate comprehensive reports based on your **{report_type.capitalize()}** results.")

        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                pdf_output = generate_report(
                    report_df_data,
                    predictions,
                    model_results,
                    st.session_state.get('risk_map_plot_buffer'),
                    kmeans_results
                )
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_output,
                    file_name=f"microplastic_risk_report_{report_type}.pdf",
                    mime="application/pdf"
                )
                st.success("PDF report generated and ready for download!")

        if st.button("Generate Excel Report (Raw Data & Results)"):
            with st.spinner("Generating Excel report..."):
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    report_df_data.to_excel(writer, sheet_name='Data_and_Results', index=True)
                
                excel_buffer.seek(0)
                st.download_button(
                    label="Download Excel Report",
                    data=excel_buffer.getvalue(),
                    file_name=f"microplastic_risk_data_and_results_{report_type}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("Excel report generated and ready for download!")

    else:
        st.warning("Please upload a dataset and run either the Classification or Clustering models before generating reports.")
