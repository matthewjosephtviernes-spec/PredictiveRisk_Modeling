import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF  # pip install fpdf
import tempfile
import os

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


# --- Helper Functions (Stubs for now) ---

def load_data(uploaded_file):
    """Loads data from CSV or Excel."""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df

def preprocess_data(df):
    """Placeholder for data cleaning and preprocessing."""
    if df is None:
        return None

    st.subheader("Data Preprocessing Steps (Applied Automatically)")
    st.info(f"Initial dataset shape: {df.shape}")

    # Example: Drop rows with any missing values (for simplicity)
    original_rows = df.shape[0]
    df.dropna(inplace=True)
    st.write(f"- Removed {original_rows - df.shape[0]} rows with missing values.")

    # Example: Drop duplicates
    original_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    st.write(f"- Removed {original_rows - df.shape[0]} duplicate rows.")

    # Example: Basic feature engineering (if 'location' column exists)
    if 'location' in df.columns:
        df['location_encoded'] = df['location'].astype('category').cat.codes
        st.write("- Encoded 'location' column.")

    st.success("Preprocessing complete!")
    return df

def train_model(df, target_column, features, model_type):
    """Placeholder for model training."""
    if df is None or target_column not in df.columns or not features:
        st.error("Cannot train model. Please check data, target, and features.")
        return None, None, None

    X = df[features]
    y = df[target_column].copy()

    # Ensure target column is numerical for classification; else convert
    if not pd.api.types.is_numeric_dtype(y):
        try:
            # Try to convert to categorical codes (generic)
            y_cat = pd.Categorical(y)
            y = y_cat.codes
            # store mapping for potential reverse mapping (useful later)
            label_mapping = {i: label for i, label in enumerate(y_cat.categories)}
            st.info(f"Converted target column '{target_column}' to numeric codes. Label mapping: {label_mapping}")
        except Exception as e:
            st.error(f"Could not convert target column '{target_column}' to numerical categories. Error: {e}")
            return None, None, None
    else:
        label_mapping = None

    # safe stratify: only if more than 1 class present
    stratify_arg = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify_arg)

    model = None
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

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

        # Build target names dynamically so classification_report doesn't crash
        unique_labels = np.unique(y_test)
        # If we have a label mapping (from categorical conversion), use readable names
        if label_mapping:
            target_names = [str(label_mapping[int(cl)]) for cl in unique_labels]
        else:
            target_names = [str(cl) for cl in unique_labels]

        # Store results for reporting (classification_report -> output_dict)
        try:
            class_rep = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        except Exception:
            # fallback: don't pass target_names to avoid possible mismatch
            class_rep = classification_report(y_test, y_pred, output_dict=True)

        st.session_state['model_report'] = {
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_rep,
            'label_mapping': label_mapping  # may be None
        }
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
    pdf.multi_cell(0, 10, "This report summarizes the analysis and predictions for microplastic pollution risk based on the provided dataset and predictive model.")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Analysis Overview", 0, 1)
    pdf.set_font("Arial", "", 12)
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

        for class_name, metrics in model_results['classification_report'].items():
            if isinstance(metrics, dict):  # For class rows like 'Low', 'Medium', 'High'
                prec = metrics.get('precision', 0.0)
                rec = metrics.get('recall', 0.0)
                f1s = metrics.get('f1-score', 0.0)
                sup = metrics.get('support', 0)
                pdf.cell(0, 5, f"  {class_name}: Precision={prec:.2f}, Recall={rec:.2f}, F1-score={f1s:.2f}, Support={sup}", 0, 1)
            else:  # For aggregated metrics like 'accuracy', 'macro avg', 'weighted avg'
                # metrics may be a float or dict; handle float
                if isinstance(metrics, (int, float)):
                    pdf.cell(0, 5, f"  {class_name}: {metrics:.2f}", 0, 1)
                else:
                    # if dict (macro avg etc.), print their values concisely
                    try:
                        pdf.cell(0, 5, f"  {class_name}: {metrics}", 0, 1)
                    except Exception:
                        pdf.cell(0, 5, f"  {class_name}: {str(metrics)}", 0, 1)
        pdf.ln(5)

    if plot_buffer:
        # plot_buffer is expected to be bytes (PNG). FPDF cannot take raw bytes directly,
        # so write to a temporary file and provide its path to pdf.image()
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(plot_buffer)
            tmp.flush()
            tmp.close()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "3. Visualizations and Risk Map", 0, 1)
            pdf.ln(2)
            # place image at current y
            y_pos = pdf.get_y()
            pdf.image(tmp.name, x=10, y=y_pos, w=190)
            pdf.ln(10)
        finally:
            # remove temp file if it exists
            try:
                if os.path.exists(tmp.name):
                    os.remove(tmp.name)
            except Exception:
                pass

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4. Identified High-Risk Zones & Mitigation Strategies", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, "Based on the predictions, high-risk zones (e.g., coordinates X, Y or specific locations A, B) are identified in areas with high industrial discharge and dense population. Suggested mitigation strategies include enhanced waste management, public awareness campaigns, and stricter industrial regulations.")
    pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')  # Return as bytes


# --- Initialize Session State ---
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


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"]
)

st.sidebar.subheader("Input Variables (Global)")
if st.session_state['processed_df'] is not None:
    all_columns = st.session_state['processed_df'].columns.tolist()
    location_col = st.sidebar.selectbox("Select Location Column", ['None'] + all_columns, key='sidebar_location')
    pollution_indicators = st.sidebar.multiselect("Select Pollution Indicators", all_columns, key='sidebar_indicators')
    temporal_col = st.sidebar.selectbox("Select Temporal Column", ['None'] + all_columns, key='sidebar_temporal')

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
    st.image("https://via.placeholder.com/700x300.png?text=Environmental+Sustainability", use_column_width=True)
    st.markdown("""
        <p style='font-size: 1.1em;'>
        This platform leverages advanced predictive analytics to assess and visualize the risk levels of microplastic pollution across various environments.
        Utilizing Streamlit for an interactive user experience and powerful data mining algorithms like Random Forest and XGBoost,
        we provide insights into pollution trends, potential high-risk zones, and actionable mitigation strategies.
        </p>
    """, unsafe_allow_html=True)

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
                st.session_state['processed_df'] = preprocess_data(st.session_state['df'].copy())

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
            fig = px.histogram(df, x=selected_column_dist, title=f'Distribution of {selected_column_dist}')
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
        if st.session_state.get('temporal_col') and st.session_state['temporal_col'] in df.columns:
            try:
                df[st.session_state['temporal_col']] = pd.to_datetime(df[st.session_state['temporal_col']])
                if st.session_state.get('pollution_indicators'):
                    trend_df = df.groupby(st.session_state['temporal_col'])[st.session_state['pollution_indicators']].mean().reset_index()
                    fig_time = px.line(trend_df, x=st.session_state['temporal_col'], y=st.session_state['pollution_indicators'],
                                       title="Pollution Indicator Trends Over Time")
                    st.plotly_chart(fig_time, use_container_width=True)
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
        # Filter out potential ID columns or non-numeric for target
        target_column_options = [col for col in target_column_options if df[col].nunique() <= 5 or not pd.api.types.is_numeric_dtype(df[col])]
        if not target_column_options:
            st.warning("No suitable target columns detected. Please check your data.")
        target_column = st.selectbox("Select Target Variable (e.g., 'Risk_Level')", target_column_options)

        # Automatically exclude target column from features list
        available_features = [col for col in df.columns if col != target_column and col != st.session_state.get('location_col') and col != st.session_state.get('temporal_col')]
        feature_columns = st.multiselect("Select Features for Prediction", available_features, default=available_features)

        model_type = st.radio("Choose Prediction Model", ["Random Forest", "XGBoost"])

        if st.button("Train Model & Generate Predictions"):
            if target_column and feature_columns:
                with st.spinner(f"Training {model_type} model..."):
                    model, X_test, y_test = train_model(df.copy(), target_column, feature_columns, model_type)
                    if model is not None:
                        st.session_state['model'] = model
                        st.session_state['test_data'] = X_test
                        st.session_state['test_labels'] = y_test

                        # Convert predictions back to labels for display if original target was categorical
                        raw_predictions = model.predict(X_test)
                        label_mapping = st.session_state.get('model_report', {}).get('label_mapping')

                        if label_mapping:
                            # label_mapping: numeric_code -> original_label
                            st.session_state['predictions'] = pd.Series(raw_predictions).map(label_mapping).values
                        else:
                            # keep numeric or original values as-is
                            st.session_state['predictions'] = raw_predictions

                        st.success("Model trained and predictions generated!")
                    else:
                        st.error("Model training failed. Please check your selections.")
            else:
                st.warning("Please select a target variable and at least one feature.")

        if st.session_state['model'] is not None and st.session_state['predictions'] is not None and st.session_state['test_data'] is not None:
            st.subheader("Prediction Results")

            # Create a dataframe for displaying predictions
            prediction_df = st.session_state['test_data'].copy() if isinstance(st.session_state['test_data'], pd.DataFrame) else pd.DataFrame(st.session_state['test_data'])
            prediction_df['True_Risk'] = st.session_state['test_labels']
            prediction_df['Predicted_Risk'] = st.session_state['predictions']

            st.write("Sample Predictions:")
            st.dataframe(prediction_df.head())

            st.subheader("Risk Map (Geographic Heatmap)")
            if st.session_state.get('location_col') and 'latitude' in df.columns and 'longitude' in df.columns:
                plot_df = df.copy()
                plot_df['predicted_risk_level'] = np.random.choice(['Low', 'Medium', 'High'], size=len(plot_df))  # Placeholder

                risk_color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                plot_df['color'] = plot_df['predicted_risk_level'].map(risk_color_map)

                fig_map = px.scatter_mapbox(plot_df,
                                            lat="latitude",
                                            lon="longitude",
                                            color="predicted_risk_level",
                                            size_max=15, zoom=3,
                                            hover_name=st.session_state['location_col'] if st.session_state['location_col'] else None,
                                            hover_data=st.session_state['pollution_indicators'] + ['predicted_risk_level'] if st.session_state.get('pollution_indicators') else ['predicted_risk_level'],
                                            color_discrete_map=risk_color_map,
                                            title="Microplastic Pollution Risk Map")
                fig_map.update_layout(mapbox_style="open-street-map")
                fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)

                # Save the plot to a buffer for PDF report (requires kaleido installed)
                try:
                    buf = BytesIO()
                    fig_map.write_image(buf, format="png", width=800, height=450, scale=2)
                    st.session_state['risk_map_plot_buffer'] = buf.getvalue()
                except Exception as e:
                    st.warning(f"Could not render map image for the PDF (kaleido might be missing). Error: {e}")
                    st.session_state['risk_map_plot_buffer'] = None
            else:
                st.warning("To view the risk map, ensure your dataset has 'latitude' and 'longitude' columns and a location column is selected in the sidebar.")
                st.session_state['risk_map_plot_buffer'] = None

            st.subheader("Model Accuracy Scorecard")
            if st.session_state.get('model_report'):
                mr = st.session_state['model_report']
                st.markdown(f"""
                - **Model Type:** {mr['model_type']}
                - **Overall Accuracy:** <span style='color: #007bff; font-weight: bold;'>{mr['accuracy']:.2f}</span>
                - **Weighted Precision:** {mr['precision']:.2f}
                - **Weighted Recall:** {mr['recall']:.2f}
                - **Weighted F1-Score:** {mr['f1_score']:.2f}
                """, unsafe_allow_html=True)
                st.write("Classification Report:")
                st.json(mr['classification_report'])
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
                    report_df = pd.DataFrame(st.session_state['test_data']).copy() if not isinstance(st.session_state['test_data'], pd.DataFrame) else st.session_state['test_data'].copy()
                    report_df['True_Risk'] = st.session_state['test_labels']
                    report_df['Predicted_Risk'] = st.session_state['predictions']

                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        report_df.to_excel(writer, sheet_name='Predictions', index=False)
                        st.session_state['processed_df'].to_excel(writer, sheet_name='Original Data', index=False)

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
