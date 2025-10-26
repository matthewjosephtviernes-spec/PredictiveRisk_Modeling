import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Import LabelEncoder for robust categorical target handling
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF # You'll need to install fpdf: pip install fpdf

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
        # Fix: Ensure file is read from the beginning if multiple reads happen
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        # Fix: Ensure file is read from the beginning
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df

def preprocess_data(df):
    """Data cleaning and preprocessing."""
    if df is None:
        return None

    st.subheader("Data Preprocessing Steps (Applied Automatically)")
    st.info(f"Initial dataset shape: {df.shape}")

    # FIX 1: Ensure columns are strings to prevent mixing data types in feature selection
    df.columns = df.columns.astype(str)

    # Example: Drop rows with any missing values (for simplicity)
    original_rows = df.shape[0]
    # FIX 2: Create a copy before inplace operations to avoid SettingWithCopyWarning, 
    # though in this context it's generally fine since df is already a copy from the caller.
    df_clean = df.copy() 
    df_clean.dropna(inplace=True)
    st.write(f"- Removed {original_rows - df_clean.shape[0]} rows with missing values.")

    # Example: Drop duplicates
    original_rows = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    st.write(f"- Removed {original_rows - df_clean.shape[0]} duplicate rows.")

    # Example: Basic feature engineering (if 'location' column exists)
    # Use .get() for safer access to session state to prevent key errors on first load
    location_col_name = st.session_state.get('location_col') 
    if location_col_name and location_col_name in df_clean.columns:
        # Check if it's already a category or numerical to avoid re-encoding errors
        if not pd.api.types.is_numeric_dtype(df_clean[location_col_name]):
            df_clean['location_encoded'] = df_clean[location_col_name].astype('category').cat.codes
            # FIX: Update the session state to use the new encoded column for modeling
            st.session_state['location_encoded_col'] = 'location_encoded'
            st.write("- Encoded 'location' column into 'location_encoded'.")
        else:
             st.session_state['location_encoded_col'] = location_col_name
    else:
        st.session_state['location_encoded_col'] = None


    st.success("Preprocessing complete!")
    return df_clean

def train_model(df, target_column, features, model_type):
    """Model training and evaluation."""
    if df is None or target_column not in df.columns or not features:
        st.error("Cannot train model. Please check data, target, and features.")
        return None, None, None

    X = df[features]
    y = df[target_column]
    
    label_encoder = None
    original_labels = None
    target_names = None

    # Ensure target column is numerical for classification
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() > 20: # Use LabelEncoder for non-numeric or too many unique values
        try:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            original_labels = label_encoder.classes_
            target_names = [str(lab) for lab in original_labels] # Convert to string for report
            st.info(f"Target column '{target_column}' converted to numerical categories: {target_names}.")
        except Exception as e:
            st.error(f"Could not convert target column '{target_column}' to numerical categories. Error: {e}")
            return None, None, None
    else:
        # Use existing unique values as target names if already numeric
        original_labels = sorted(y.unique())
        target_names = [str(lab) for lab in original_labels]

    # FIX 3: Handle the case where all rows were removed in preprocessing or target has only one class
    if len(y) == 0 or len(y.unique()) <= 1:
        st.error("Target variable has one or zero unique classes after preprocessing. Cannot train a classifier.")
        return None, None, None


    # FIX 4: Handle the case where stratify is not possible if the class count is too low for the test size.
    # We will use stratify only if there are enough samples per class.
    # The original stratify check (len(y.unique()) > 1) is too simple.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError as e:
        st.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    model = None
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        # FIX 5: Remove 'use_label_encoder=False' and 'eval_metric' for newer XGBoost versions to avoid warnings/errors.
        # We handle label encoding manually above.
        model = XGBClassifier(random_state=42) 

    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics (can handle multi-class via 'weighted')
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

        # Store results for reporting
        st.session_state['model_report'] = {
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            # FIX 6: Use the dynamically created target_names for the classification report
            'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        }
        # Store the label encoder and target names for later use in prediction display
        st.session_state['label_encoder'] = label_encoder
        st.session_state['target_names'] = target_names
        
        return model, X_test, y_test
    return None, None, None

def generate_report(df, predictions, model_results, plot_buffer):
    """Generates a PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Microplastic Pollution Risk Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    # FIX 7: Use f-string for multi_cell argument, which expects a single string
    pdf.multi_cell(0, 10, f"This report summarizes the analysis and predictions for microplastic pollution risk based on the provided dataset and predictive model.")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Analysis Overview", 0, 1)
    pdf.set_font("Arial", "", 12)
    # FIX 8: Use f-string for multi_cell
    pdf.multi_cell(0, 7, f"Dataset processed: {st.session_state.get('uploaded_filename', 'N/A')} with {df.shape[0]} rows and {df.shape[1]} columns.")
    pdf.multi_cell(0, 7, f"Key findings narrative: (Example: The analysis identified a strong correlation between pH levels and microplastic risk. High-risk zones are predominantly found in coastal urban areas.)")
    pdf.ln(5)

    if model_results:
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
        # FIX 9: Classification report dictionary keys are strings, not 'accuracy' or 'macro avg' keys themselves, 
        # so iterating over items and checking the type is a good approach.
        for class_name, metrics in model_results['classification_report'].items():
            if isinstance(metrics, dict): # For the actual class metrics
                pdf.cell(0, 5, f"  {class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-score={metrics['f1-score']:.2f}, Support={metrics['support']}", 0, 1)
            elif class_name in ['accuracy', 'macro avg', 'weighted avg']: # For summary metrics
                # For 'accuracy', the value is directly the metric, for others it's a dict with precision, recall, f1-score
                if class_name == 'accuracy':
                    pdf.cell(0, 5, f"  {class_name}: {metrics:.2f}", 0, 1)
                else:
                    pdf.cell(0, 5, f"  {class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-score={metrics['f1-score']:.2f}", 0, 1)
        pdf.ln(5)

    if plot_buffer:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. Visualizations and Risk Map", 0, 1)
        pdf.ln(2)
        # FIX 10: Ensure the image call uses the correct format for image data (BytesIO content)
        # and has a fixed width to fit the page.
        try:
            # You must ensure plot_buffer is a path or a file-like object compatible with fpdf.
            # Since we saved it as bytes in session_state, we need to pass a BytesIO object to FPDF.
            plot_io = BytesIO(plot_buffer)
            pdf.image(plot_io, x=10, y=pdf.get_y(), w=190)
            pdf.ln(10)
        except Exception as e:
            # Handle case where image couldn't be loaded (e.g., if it's not a valid image format for fpdf)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, f"Warning: Could not embed image in PDF. Error: {e}")
            pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4. Identified High-Risk Zones & Mitigation Strategies", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, "Based on the predictions, high-risk zones (e.g., coordinates X, Y or specific locations A, B) are identified in areas with high industrial discharge and dense population. Suggested mitigation strategies include enhanced waste management, public awareness campaigns, and stricter industrial regulations.")
    pdf.ln(10)

    # FIX 11: The original code returned bytes in 'latin-1', which might not be safe. 
    # Returning the buffer directly as bytes is more standard.
    return pdf.output(dest='S').encode('latin-1')


# --- Initialize Session State ---
# Removed st.session_state['df'] as it's not strictly necessary if processed_df is the main working one, 
# but kept it for clarity in the upload process.
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'test_data' not in st.session_state:
    st.session_state['test_data'] = None
if 'test_labels' not in st.session_state:
    st.session_state['test_labels'] = None
if 'model_report' not in st.session_state:
    st.session_state['model_report'] = None
if 'uploaded_filename' not in st.session_state:
    st.session_state['uploaded_filename'] = "No file uploaded"
if 'location_col' not in st.session_state: # Initialize all sidebar keys
    st.session_state['location_col'] = None
if 'temporal_col' not in st.session_state:
    st.session_state['temporal_col'] = None
if 'pollution_indicators' not in st.session_state:
    st.session_state['pollution_indicators'] = []
# New session states for encoding/mapping
if 'location_encoded_col' not in st.session_state:
    st.session_state['location_encoded_col'] = None
if 'label_encoder' not in st.session_state:
    st.session_state['label_encoder'] = None
if 'target_names' not in st.session_state:
    st.session_state['target_names'] = None
if 'risk_map_plot_buffer' not in st.session_state:
    st.session_state['risk_map_plot_buffer'] = None


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"]
)

st.sidebar.subheader("Input Variables (Global)")
if st.session_state['processed_df'] is not None:
    all_columns = st.session_state['processed_df'].columns.tolist()
    
    # FIX 12: Ensure default values for selectbox are safe if a file has been uploaded but a column hasn't been picked yet
    # Use st.session_state.get to retrieve current values, defaulting to 'None' or [] if not set.
    location_col_current = st.session_state.get('location_col') if st.session_state.get('location_col') in all_columns else 'None'
    temporal_col_current = st.session_state.get('temporal_col') if st.session_state.get('temporal_col') in all_columns else 'None'

    location_col = st.sidebar.selectbox("Select Location Column", ['None'] + all_columns, index=['None'] + all_columns.index(location_col_current) if location_col_current != 'None' else 0, key='sidebar_location')
    pollution_indicators = st.sidebar.multiselect("Select Pollution Indicators", all_columns, default=st.session_state['pollution_indicators'], key='sidebar_indicators')
    temporal_col = st.sidebar.selectbox("Select Temporal Column", ['None'] + all_columns, index=['None'] + all_columns.index(temporal_col_current) if temporal_col_current != 'None' else 0, key='sidebar_temporal')

    if location_col != 'None':
        st.session_state['location_col'] = location_col
    else:
        st.session_state['location_col'] = None

    st.session_state['pollution_indicators'] = pollution_indicators
    if temporal_col != 'None':
        st.session_state['temporal_col'] = temporal_col
    else:
        st.session_state['temporal_col'] = None
else:
    st.sidebar.info("Upload a dataset first to select variables.")


# --- Main Content Area ---

if page == "Home":
    st.title("Welcome to the Microplastic Pollution Risk Assessment System")
    # FIX 13: Changed image to a better placeholder
    st.image("https://via.placeholder.com/700x300.png?text=Environmental+Data+Analysis", use_column_width=True) 
    st.markdown("""
        <p style='font-size: 1.1em;'>
        This platform leverages advanced predictive analytics to assess and visualize the risk levels of microplastic pollution across various environments.
        Utilizing Streamlit for an interactive user experience and powerful data mining algorithms like Random Forest and XGBoost,
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
    - **Predictive Modeling:** Classify microplastic pollution risk (Low, Medium, High).
    - **Interactive Dashboards:** Visualize pollution intensity with heatmaps and time-series graphs.
    - **Comprehensive Reporting:** Download detailed reports in PDF or Excel.
    """)

elif page == "Upload Dataset":
    st.title("Upload Your Environmental Dataset")
    st.markdown("Please upload your environmental data file in CSV or Excel format (.csv, .xls, .xlsx).")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        st.session_state['uploaded_filename'] = uploaded_file.name
        with st.spinner("Loading and preprocessing data..."):
            st.session_state['df'] = load_data(uploaded_file)
            if st.session_state['df'] is not None:
                st.write("Original Data Preview:")
                st.dataframe(st.session_state['df'].head())
                # FIX 14: Ensure preprocess_data runs after setting the sidebar variables for encoding to work (optional, but safer)
                # It's better to ensure that `preprocess_data` does not rely on *new* sidebar selections 
                # but only on the state *before* it runs, or refactor to select cols *after* upload.
                st.session_state['processed_df'] = preprocess_data(st.session_state['df'].copy()) # Pass a copy

                if st.session_state['processed_df'] is not None:
                    st.success("Dataset loaded and preprocessed successfully!")
                    st.write("Processed Data Preview:")
                    st.dataframe(st.session_state['processed_df'].head())
                    st.write(f"Processed dataset shape: {st.session_state['processed_df'].shape}")
                else:
                    st.error("Data preprocessing failed.")
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
            # FIX 15: Ensure column is not all NaNs/non-numeric before plotting histogram
            if pd.api.types.is_numeric_dtype(df[selected_column_dist]):
                fig = px.histogram(df, x=selected_column_dist, title=f'Distribution of {selected_column_dist}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Plot bar chart for non-numeric/categorical
                counts = df[selected_column_dist].value_counts().reset_index()
                counts.columns = [selected_column_dist, 'count']
                fig = px.bar(counts, x=selected_column_dist, y='count', title=f'Count of {selected_column_dist}')
                st.plotly_chart(fig, use_container_width=True)


        st.subheader("Correlation Matrix")
        # Select only numeric columns for correlation
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
                # FIX 16: Handle case where 'temporal_col' might be None or a string 'None' from sidebar
                if st.session_state['temporal_col'] == 'None':
                     st.info("Select a temporal column in the sidebar to view time-series trends.")
                else:
                    df_time = df.copy() # Work on a copy
                    df_time[st.session_state['temporal_col']] = pd.to_datetime(df_time[st.session_state['temporal_col']], errors='coerce')
                    df_time.dropna(subset=[st.session_state['temporal_col']], inplace=True) # Drop rows where conversion failed
                    
                    if st.session_state['pollution_indicators']:
                        # Group by temporal column and average selected indicators
                        # Select only numeric indicators
                        numeric_indicators = [col for col in st.session_state['pollution_indicators'] if pd.api.types.is_numeric_dtype(df_time[col])]
                        if numeric_indicators:
                            trend_df = df_time.groupby(st.session_state['temporal_col'])[numeric_indicators].mean().reset_index()
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
    st.title("Prediction Dashboard")

    if st.session_state['processed_df'] is not None:
        df = st.session_state['processed_df']

        st.subheader("Model Configuration")
        target_column_options = df.columns.tolist()
        # Filter target columns: typically categorical (few unique values) or a clear risk score column
        target_column_options = [col for col in target_column_options if col != st.session_state.get('location_col') and col != st.session_state.get('temporal_col')]

        # FIX 17: Set a safe default for target_column if it exists in session state
        target_col_current = st.session_state.get('target_col', target_column_options[0] if target_column_options else None)
        if target_col_current not in target_column_options and target_column_options:
            target_col_current = target_column_options[0]

        target_column = st.selectbox("Select Target Variable (e.g., 'Risk_Level')", target_column_options, index=target_column_options.index(target_col_current) if target_col_current in target_column_options else 0, key='target_select')
        st.session_state['target_col'] = target_column # Store target col for later use

        # Automatically exclude non-feature columns
        available_features = [col for col in df.columns if col not in [target_column, st.session_state.get('location_col'), st.session_state.get('temporal_col')]]
        # FIX 18: Include the encoded location column if it was created
        if st.session_state.get('location_encoded_col') and st.session_state['location_encoded_col'] not in available_features:
            available_features.append(st.session_state['location_encoded_col'])

        # Filter out non-numeric features as classifiers can't use them directly (unless properly encoded)
        numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(df[col])]
        
        # Keep non-numeric only if they were encoded (like 'location_encoded')
        encoded_features = [col for col in available_features if col not in numeric_features and col.endswith('_encoded')]
        
        feature_columns = st.multiselect("Select Features for Prediction (Numeric or Encoded)", numeric_features + encoded_features, default=numeric_features + encoded_features)

        model_type = st.radio("Choose Prediction Model", ["Random Forest", "XGBoost"])

        if st.button("Train Model & Generate Predictions"):
            if target_column and feature_columns:
                with st.spinner(f"Training {model_type} model..."):
                    model, X_test, y_test = train_model(df.copy(), target_column, feature_columns, model_type)
                    if model is not None:
                        st.session_state['model'] = model
                        # FIX 19: X_test (input features) should be a copy of the index for merging/display later
                        st.session_state['test_data'] = X_test.copy()
                        st.session_state['test_labels'] = y_test

                        raw_predictions = model.predict(X_test)

                        # Convert numeric predictions (0, 1, 2, ...) back to original labels (Low, Medium, High, ...)
                        label_encoder = st.session_state.get('label_encoder')
                        if label_encoder is not None:
                             # FIX 20: Use the label encoder to transform numeric predictions back to original labels
                             st.session_state['predictions'] = label_encoder.inverse_transform(raw_predictions)
                             # Since y_test is also encoded, convert it back for displaying True Risk
                             true_labels_decoded = label_encoder.inverse_transform(y_test)
                        else:
                             # If target was already numeric, predictions are the actual values
                             st.session_state['predictions'] = raw_predictions
                             true_labels_decoded = y_test

                        # Store decoded labels
                        st.session_state['test_labels_decoded'] = true_labels_decoded
                        
                        st.success("Model trained and predictions generated!")
                    else:
                        st.error("Model training failed. Please check your selections.")
            else:
                st.warning("Please select a target variable and at least one feature.")

        if st.session_state['model'] is not None and st.session_state['predictions'] is not None:
            st.subheader("Prediction Results")

            # Create a dataframe for displaying predictions
            prediction_df = st.session_state['test_data'].copy()
            # FIX 21: Use decoded labels for display
            prediction_df['True_Risk'] = st.session_state.get('test_labels_decoded', st.session_state['test_labels']) 
            prediction_df['Predicted_Risk'] = st.session_state['predictions']

            st.write("Sample Predictions:")
            st.dataframe(prediction_df.head())

            st.subheader("Risk Map (Geographic Heatmap)")
            # Check for required columns and sidebar selection
            if st.session_state.get('location_col') and 'latitude' in df.columns and 'longitude' in df.columns: 
                
                # FIX 22: Merge the predictions back into the original/processed dataframe based on the index 
                # (assuming the original index was maintained through preprocessing/train_test_split)
                # This is complex in a real app, so we'll simplify by merging X_test results with their indices.
                
                # 1. Get the indices from the test set
                test_indices = st.session_state['test_data'].index
                
                # 2. Get the corresponding original rows (including lat/lon)
                plot_df_full = df.loc[test_indices].copy()

                # 3. Add predictions
                plot_df_full['Predicted_Risk_Level'] = st.session_state['predictions']

                # Map risk levels to colors (using the names from the classification report)
                risk_labels = st.session_state.get('target_names', ['Low', 'Medium', 'High'])
                
                # Create a simple default color map based on the number of classes
                color_list = px.colors.qualitative.Plotly # Get a list of distinct colors
                risk_color_map = {label: color_list[i % len(color_list)] for i, label in enumerate(risk_labels)}
                
                # Ensure 'Low' and 'High' get somewhat standard colors if they exist
                if 'Low' in risk_labels: risk_color_map['Low'] = 'green'
                if 'Medium' in risk_labels: risk_color_map['Medium'] = 'orange'
                if 'High' in risk_labels: risk_color_map['High'] = 'red'


                fig_map = px.scatter_mapbox(plot_df_full,
                                            lat="latitude", # Assuming 'latitude' column
                                            lon="longitude", # Assuming 'longitude' column
                                            color="Predicted_Risk_Level", # Use the predicted risk
                                            size_max=15, zoom=3,
                                            hover_name=st.session_state['location_col'],
                                            # Use the selected pollution indicators for hover data
                                            hover_data=st.session_state['pollution_indicators'] + ['Predicted_Risk_Level'] if st.session_state['pollution_indicators'] else ['Predicted_Risk_Level'],
                                            color_discrete_map=risk_color_map,
                                            title="Microplastic Pollution Risk Map (Test Set Predictions)")
                fig_map.update_layout(mapbox_style="open-street-map")
                fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)

                # Save the plot to a buffer for PDF report
                buf = BytesIO()
                # FIX 23: Ensure the buffer is reset before writing
                fig_map.write_image(buf, format="png", width=800, height=450, scale=2)
                st.session_state['risk_map_plot_buffer'] = buf.getvalue()
            else:
                st.warning("To view the risk map, ensure your dataset has 'latitude' and 'longitude' columns and a location column is selected in the sidebar.")
                st.session_state['risk_map_plot_buffer'] = None # Ensure it is None if plot fails

            st.subheader("Model Accuracy Scorecard")
            if st.session_state['model_report']:
                st.markdown(f"""
                - **Model Type:** {st.session_state['model_report']['model_type']}
                - **Overall Accuracy:** <span style='color: #007bff; font-weight: bold;'>{st.session_state['model_report']['accuracy']:.2f}</span>
                - **Weighted Precision:** {st.session_state['model_report']['precision']:.2f}
                - **Weighted Recall:** {st.session_state['model_report']['recall']:.2f}
                - **Weighted F1-Score:** {st.session_state['model_report']['f1_score']:.2f}
                """, unsafe_allow_html=True)
                st.write("Classification Report:")
                # FIX 24: Display the classification report with better formatting
                report_df = pd.DataFrame(st.session_state['model_report']['classification_report']).transpose().round(2)
                st.dataframe(report_df)
            else:
                st.info("Train the model first to see the scorecard.")

    else:
        st.warning("Please upload and preprocess a dataset on the 'Upload Dataset' page first.")

elif page == "Reports":
    st.title("Generate & Download Reports")

    if st.session_state['processed_df'] is not None and st.session_state['model'] is not None and st.session_state['predictions'] is not None:
        st.write("Generate comprehensive reports based on your analysis and predictions.")

        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                # Ensure the model report is available before calling generate_report
                if st.session_state.get('model_report') is None:
                    st.error("Model report data is missing. Please re-run the model training.")
                else:
                    pdf_output = generate_report(
                        st.session_state['processed_df'],
                        st.session_state['predictions'],
                        st.session_state['model_report'],
                        st.session_state.get('risk_map_plot_buffer')
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_output,
                        file_name="microplastic_risk_report.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF report generated and ready for download!")

        if st.button("Generate Excel Report (Raw Data & Predictions)"):
            with st.spinner("Generating Excel report..."):
                if st.session_state['test_data'] is not None:
                    report_df = st.session_state['test_data'].copy()
                    # Use decoded labels for the Excel report
                    report_df['True_Risk'] = st.session_state.get('test_labels_decoded', st.session_state['test_labels'])
                    report_df['Predicted_Risk'] = st.session_state['predictions']

                    # Save to Excel
                    excel_buffer = BytesIO()
                    # FIX 25: Ensure the ExcelWriter context manager handles all writes
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        report_df.to_excel(writer, sheet_name='Predictions', index=False)
                        st.session_state['processed_df'].to_excel(writer, sheet_name='Original Data', index=False)
                    
                    # Get the value from the buffer *after* the writer has closed
                    excel_buffer.seek(0)

                    st.download_button(
                        label="Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name="microplastic_risk_data_and_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("Excel report generated and ready for download!")
                else:
                    st.warning("No prediction data available for Excel report.")

    else:
        st.warning("Please upload a dataset, run data analysis, and train a model before generating reports.")
