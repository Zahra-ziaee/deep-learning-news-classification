import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import FIGURES_DIR


def plot_class_distribution(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = df["category"].value_counts()

    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.index, class_counts.values)
    plt.title("News Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_training_curves(history) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Training Accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
    plt.title("BiLSTM Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "training_accuracy.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Training Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.title("BiLSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "training_loss.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_confusion_matrix(y_true, y_pred) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    labels = sorted(pd.Series(y_true).unique())

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, xticks_rotation=30)
    plt.title("BiLSTM Confusion Matrix")
    plt.tight_layout()

    output_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")