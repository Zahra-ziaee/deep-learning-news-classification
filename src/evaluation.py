import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.config import CLASSIFICATION_REPORT_FILE, METRICS_FILE, RESULTS_DIR


def evaluate_classification_model(model_name: str, y_true, y_pred) -> dict:
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_models(
    baseline_model,
    deep_model,
    X_test_text,
    X_test_padded,
    y_test_labels,
    y_test_encoded,
    label_encoder,
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_predictions = baseline_model.predict(X_test_text)

    deep_probabilities = deep_model.predict(X_test_padded)
    deep_predictions_encoded = np.argmax(deep_probabilities, axis=1)
    deep_predictions = label_encoder.inverse_transform(deep_predictions_encoded)

    metrics = [
        evaluate_classification_model(
            model_name="TF-IDF + Logistic Regression",
            y_true=y_test_labels,
            y_pred=baseline_predictions,
        ),
        evaluate_classification_model(
            model_name="BiLSTM Deep Learning Model",
            y_true=y_test_labels,
            y_pred=deep_predictions,
        ),
    ]

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_FILE, index=False)

    report_dict = classification_report(
        y_test_labels,
        deep_predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_FILE)

    print("\nModel comparison metrics:")
    print(metrics_df)

    print(f"\nMetrics saved to: {METRICS_FILE}")
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_FILE}")

    return metrics_df, baseline_predictions, deep_predictions