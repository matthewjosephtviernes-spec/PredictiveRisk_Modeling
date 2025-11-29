import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns


# ============ HELPER FUNCTIONS ============

def clean_dash_chars(s: str) -> str:
    """
    Replace weird dash characters (like , –, —) with a standard '-'.
    """
    if not isinstance(s, str):
        return s
    return s.replace("", "-").replace("–", "-").replace("—", "-")


def parse_range_to_mean(value):
    """
    Parse values like '0.1-5.0' or '1.3-1.4' to their mean.
    If just a single number, return it.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value = clean_dash_chars(str(value))
    # Keep digits, dots, minus and separators
    # Split on '-' if exists
    parts = re.split(r"-", value)

    nums = []
    for p in parts:
        # Remove non numeric/decimal chars
        p_clean = re.sub(r"[^0-9.]", "", p)
        if p_clean != "":
            try:
                nums.append(float(p_clean))
            except ValueError:
                pass

    if len(nums) == 0:
        return np.nan
    elif len(nums) == 1:
        return nums[0]
    else:
        return float(np.mean(nums))


def extract_numeric(value):
    """
    Extract first numeric from a string (e.g. '33 ppt' -> 33.0).
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def expand_multi_label_column(df, column_name):
    """
    Expand a column like 'Lines, Fragments, Films' into binary indicator columns.
    """
    if column_name not in df.columns:
        return df

    # Get all unique labels
    labels_set = set()
    for val in df[column_name].dropna():
        parts = [p.strip() for p in str(val).split(",")]
        labels_set.update(p for p in parts if p != "")

    for label in labels_set:
        col_label = re.sub(r"\s+", "_", label)  # replace spaces
        new_col_name = f"{column_name}_{col_label}"
        df[new_col_name] = df[column_name].apply(
            lambda x: 1 if isinstance(x, str) and label in [p.strip() for p in x.split(",")] else 0
        )

    # Drop original multi-label column
    df = df.drop(columns=[column_name])
    return df


def preprocess_dataframe(df):
    """
    Apply all necessary cleaning & feature engineering to your dataset.
    """

    df = df.copy()

    # Clean dash-like chars in Microplastic_Size_mm and Density
    if "Microplastic_Size_mm" in df.columns:
        df["Microplastic_Size_mm_clean"] = df["Microplastic_Size_mm"].apply(parse_range_to_mean)

    if "Density" in df.columns:
        df["Density_clean"] = df["Density"].apply(parse_range_to_mean)

    # Extract numeric salinity
    if "Salinity" in df.columns:
        df["Salinity_clean"] = df["Salinity"].apply(extract_numeric)

    # Convert numeric-like columns
    numeric_cols = ["Latitude", "Longitude", "pH", "MP_Count_per_L", "Risk_Score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Expand multi-label columns: Shape, Polymer_Type
    if "Shape" in df.columns:
        df = expand_multi_label_column(df, "Shape")

    if "Polymer_Type" in df.columns:
        df = expand_multi_label_column(df, "Polymer_Type")

    # Drop unnecessary columns (that are not useful for prediction)
    for col in ["Author"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Optional: Risk_Type might be dropped if it is constant
    if "Risk_Type" in df.columns:
        if df["Risk_Type"].nunique() <= 1:
            df = df.drop(columns=["Risk_Type"])

    return df


def build_model_pipeline(model_name, numeric_features, categorical_features):
    """
    Create a sklearn Pipeline with preprocessing + chosen classifier.
    """

    # Preprocess categorical features with OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Choose model
    if model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(random_state=42)
    elif model_name == "Logistic Regression":
        clf = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            solver="lbfgs"
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf)
    ])

    return pipe


# ============ STREAMLIT APP ============

def main():
    st.set_page_config(page_title="Microplastic Risk Classification", layout="wide")

    st.title("🌊 Microplastic Pollution Risk Classification")
    st.write(
        """
        This app implements your thesis framework: a **predictive risk modeling** 
        system for microplastic pollution using **classification models**.
        Upload your dataset, preprocess it, train models, and visualize the results.
        """
    )

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

    target_column = "Risk_Level"  # Based on your dataset

    model_choice = st.sidebar.selectbox(
        "Select Classification Model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    )

    cv_folds = st.sidebar.slider("K-Fold Cross-Validation (k)", min_value=3, max_value=10, value=5, step=1)

    if uploaded_file is None:
        st.info("⬆️ Please upload your dataset (.csv) to begin.")
        return

    # Read data
    try:
        df_raw = pd.read_csv(uploaded_file, encoding="latin1")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    st.subheader("📁 Raw Dataset Preview")
    st.dataframe(df_raw.head())

    st.write(f"**Rows:** {df_raw.shape[0]} | **Columns:** {df_raw.shape[1]}")

    if target_column not in df_raw.columns:
        st.error(f"Target column '{target_column}' not found in dataset.")
        return

    # Preprocessing
    st.subheader("🧹 Preprocessing")
    df = preprocess_dataframe(df_raw)

    st.write("Preview of preprocessed data:")
    st.dataframe(df.head())

    # Prepare X, y
    df_model = df.dropna(subset=[target_column])
    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    st.write(f"After preprocessing and dropping missing target rows: **{X.shape[0]} samples**, **{X.shape[1]} features**")

    # Identify numeric & categorical features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    st.write("**Numeric Features:**", numeric_features)
    st.write("**Categorical Features:**", categorical_features)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/Test Split
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    st.subheader("🧪 Train / Test Split")
    st.write(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Build pipeline
    pipe = build_model_pipeline(model_choice, numeric_features, categorical_features)

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            pipe.fit(X_train, y_train)

        st.success("Model training complete!")

        # Predictions
        y_pred = pipe.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.subheader("📊 Test Set Performance")
        st.write(f"**Accuracy:** {acc:.4f}")

        # Classification report
        st.write("**Classification Report:**")
        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))

        # Confusion Matrix
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

        # Cross-validation
        st.subheader(f"🔁 {cv_folds}-Fold Cross-Validation on Full Dataset")
        with st.spinner("Running cross-validation..."):
            cv_scores = cross_val_score(pipe, X, y_encoded, cv=cv_folds, scoring="accuracy")

        st.write(f"**CV Mean Accuracy:** {cv_scores.mean():.4f}")
        st.write(f"**CV Std Dev:** {cv_scores.std():.4f}")
        st.write("Fold-wise accuracies:", np.round(cv_scores, 4))

        fig_cv, ax_cv = plt.subplots()
        ax_cv.boxplot(cv_scores, vert=False)
        ax_cv.set_title(f"{cv_folds}-Fold CV Accuracy")
        ax_cv.set_xlabel("Accuracy")
        st.pyplot(fig_cv)

        # Feature importance (only for tree-based models)
        st.subheader("🌟 Feature Importance (Tree-based models only)")

        try:
            # Extract trained model from pipeline
            model = pipe.named_steps["model"]

            if hasattr(model, "feature_importances_"):
                # Need feature names after preprocessing
                preprocessor = pipe.named_steps["preprocess"]

                # Get feature names from ColumnTransformer
                cat_ohe = preprocessor.named_transformers_["cat"]
                cat_feature_names = cat_ohe.get_feature_names_out(categorical_features)
                feature_names = np.concatenate(
                    [numeric_features, cat_feature_names]
                )

                importances = model.feature_importances_
                feat_imp = pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances
                }).sort_values("importance", ascending=False).head(20)

                st.write("Top 20 most important features:")
                st.dataframe(feat_imp)

                fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
                ax_imp.barh(feat_imp["feature"], feat_imp["importance"])
                ax_imp.invert_yaxis()
                ax_imp.set_xlabel("Importance")
                ax_imp.set_ylabel("Feature")
                ax_imp.set_title("Top 20 Feature Importances")
                st.pyplot(fig_imp)
            else:
                st.info("Selected model does not provide feature_importances_. Try Random Forest or Gradient Boosting.")
        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")


if __name__ == "__main__":
    main()
