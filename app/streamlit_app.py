import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSIFICATION_REPORT_FILE, FIGURES_DIR, METRICS_FILE
from src.predict import predict_news_category


def load_metrics():
    metrics = pd.read_csv(METRICS_FILE)
    report = pd.read_csv(CLASSIFICATION_REPORT_FILE)
    return metrics, report


def main():
    st.set_page_config(
        page_title="Deep Learning News Classification",
        layout="wide",
    )

    st.title("📰 Deep Learning News Classification")

    st.write(
        "This app predicts the category of a news article using a Deep Learning "
        "BiLSTM text classification model. The project also compares the deep "
        "learning model with a TF-IDF + Logistic Regression baseline."
    )

    st.divider()

    try:
        metrics, report = load_metrics()

    except FileNotFoundError:
        st.error("Model outputs were not found. Please run the pipeline first:")
        st.code("python main.py")
        return

    st.subheader("Model Comparison")

    st.dataframe(
        metrics,
        hide_index=True,
        width="stretch",
    )

    st.divider()

    st.subheader("Predict News Category")

    example_text = (
        "The baseball team won the championship after a strong performance "
        "from the pitcher and several home runs."
    )

    text_input = st.text_area(
        "Enter a news article or short text",
        value=example_text,
        height=180,
    )

    if st.button("Predict Category"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            return

        result = predict_news_category(text_input)

        col1, col2 = st.columns(2)

        col1.metric("Predicted Category", result["predicted_category"])
        col2.metric("Confidence", result["confidence_percent"])

        st.markdown("### Class Probabilities")
        probabilities_df = pd.DataFrame(
            {
                "category": list(result["class_probabilities"].keys()),
                "probability": list(result["class_probabilities"].values()),
            }
        )

        st.dataframe(
            probabilities_df,
            hide_index=True,
            width="stretch",
        )

    st.divider()

    st.subheader("Model Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        class_dist_path = FIGURES_DIR / "class_distribution.png"
        if class_dist_path.exists():
            st.image(str(class_dist_path), caption="Class Distribution")

    with col2:
        confusion_path = FIGURES_DIR / "confusion_matrix.png"
        if confusion_path.exists():
            st.image(str(confusion_path), caption="BiLSTM Confusion Matrix")

    col3, col4 = st.columns(2)

    with col3:
        accuracy_path = FIGURES_DIR / "training_accuracy.png"
        if accuracy_path.exists():
            st.image(str(accuracy_path), caption="Training Accuracy")

    with col4:
        loss_path = FIGURES_DIR / "training_loss.png"
        if loss_path.exists():
            st.image(str(loss_path), caption="Training Loss")

    st.divider()

    st.subheader("Project Components")

    st.markdown(
        """
        - Text preprocessing and cleaning
        - TF-IDF + Logistic Regression baseline
        - Tokenization and sequence padding
        - Embedding layer and BiLSTM deep learning model
        - Multi-class classification evaluation
        - Confusion matrix and training curves
        - Streamlit prediction demo
        """
    )


if __name__ == "__main__":
    main()