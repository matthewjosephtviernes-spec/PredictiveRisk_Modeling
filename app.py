import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Microplastic Risk Analysis — Full Preprocessing & Modeling App (Fixed Version)")

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel Dataset", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Outlier Handling
    # -----------------------------
    num_cols = ["MP_Count_per_L", "Risk_Score", "Microplastic_Size_mm_midpoint", "Density_midpoint"]
    for col in num_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)

    # -----------------------------
    # Skewness Transformation
    # -----------------------------
    for col in num_cols:
        if col in df.columns and df[col].skew() > 1:
            df[col] = np.log1p(df[col])

    # -----------------------------
    # Encoding Categorical Variables
    # -----------------------------
    cat_cols = ["Location", "Shape", "Polymer_Type", "pH", "Salinity", "Industrial_Activity", 
                "Population_Density", "Risk_Type", "Risk_Level", "Author"]

    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # -----------------------------
    # Feature Scaling
    # -----------------------------
    scaler = StandardScaler()
    for col in num_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])

    st.subheader("Preprocessed Dataset")
    st.dataframe(df.head())

    # -----------------------------
    # Risk Score Distribution
    # -----------------------------
    st.subheader("Risk Score Distribution")
    if 'Risk_Score' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['Risk_Score'], kde=True, ax=ax)
        st.pyplot(fig)

    # -----------------------------
    # Risk Score vs MP Count
    # -----------------------------
    if 'Risk_Score' in df.columns and 'MP_Count_per_L' in df.columns:
        st.subheader("Risk Score vs MP_Count_per_L")
        fig, ax = plt.subplots()
        ax.scatter(df['Risk_Score'], df['MP_Count_per_L'])
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("MP Count per L")
        st.pyplot(fig)

    # -----------------------------
    # Risk Score by Risk Level
    # -----------------------------
    if 'Risk_Level' in df.columns:
        st.subheader("Risk Score by Risk Level")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['Risk_Level'], y=df['Risk_Score'], ax=ax)
        st.pyplot(fig)

    # -----------------------------
    # Modeling
    # -----------------------------
    if "Risk_Type" in df.columns and "Risk_Level" in df.columns:
        X = df.drop(columns=["Risk_Type", "Risk_Level"])
        y_type = df['Risk_Type']
        y_level = df['Risk_Level']

        X_train, X_test, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)
        _, _, y_train_level, y_test_level = train_test_split(X, y_level, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        st.subheader("Model Evaluation — Risk_Type & Risk_Level")
        results_type = {}
        results_level = {}

        for name, model in models.items():
            # Risk Type
            model.fit(X_train, y_train_type)
            preds_type = model.predict(X_test)
            results_type[name] = [
                accuracy_score(y_test_type, preds_type),
                precision_score(y_test_type, preds_type, average='weighted'),
                recall_score(y_test_type, preds_type, average='weighted'),
                f1_score(y_test_type, preds_type, average='weighted')
            ]

            # Risk Level
            model.fit(X_train, y_train_level)
            preds_level = model.predict(X_test)
            results_level[name] = [
                accuracy_score(y_test_level, preds_level),
                precision_score(y_test_level, preds_level, average='weighted'),
                recall_score(y_test_level, preds_level, average='weighted'),
                f1_score(y_test_level, preds_level, average='weighted')
            ]

        st.write("Risk_Type Performance:")
        st.dataframe(pd.DataFrame(results_type, index=["Accuracy","Precision","Recall","F1-Score"]))

        st.write("Risk_Level Performance:")
        st.dataframe(pd.DataFrame(results_level, index=["Accuracy","Precision","Recall","F1-Score"]))

        # -----------------------------
        # K-Fold Validation
        # -----------------------------
        st.subheader("K-Fold Cross Validation")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {}

        for name, model in models.items():
            scores = cross_val_score(model, X, y_type, cv=kf, scoring='accuracy')
            cv_scores[name] = scores

        st.write("Cross Validation Accuracy (Risk_Type):")
        st.dataframe(pd.DataFrame(cv_scores))

        # -----------------------------
        # Model Performance Visualization
        # -----------------------------
        st.subheader("Model Performance Visualization")
        perf_df = pd.DataFrame(results_type, index=["Accuracy","Precision","Recall","F1-Score"])
        fig, ax = plt.subplots()
        perf_df.plot(kind='bar', ax=ax)
        st.pyplot(fig)
