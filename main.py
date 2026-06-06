from src.baseline_model import train_baseline_model
from src.data_loader import load_news_dataset, print_dataset_summary, save_processed_dataset
from src.deep_model import train_deep_model
from src.evaluation import evaluate_models
from src.preprocessing import (
    encode_labels,
    prepare_text_data,
    split_dataset,
    tokenize_text,
)
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_training_curves,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    print_section("Deep Learning News Classification Pipeline")

    print_section("Loading Dataset")
    raw_df = load_news_dataset()
    print_dataset_summary(raw_df)

    print_section("Preprocessing")
    processed_df = prepare_text_data(raw_df)
    save_processed_dataset(processed_df)
    print_dataset_summary(processed_df)

    print_section("Train/Test Split")
    X_train, X_test, y_train, y_test = split_dataset(processed_df)

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    print_section("Baseline Model")
    baseline_model = train_baseline_model(X_train, y_train)

    print_section("Deep Learning Preprocessing")
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)
    X_train_padded, X_test_padded, tokenizer = tokenize_text(X_train, X_test)

    print_section("Deep Learning Model")
    num_classes = len(label_encoder.classes_)
    deep_model, history = train_deep_model(
        X_train_padded=X_train_padded,
        y_train_encoded=y_train_encoded,
        X_test_padded=X_test_padded,
        y_test_encoded=y_test_encoded,
        num_classes=num_classes,
    )

    print_section("Evaluation")
    metrics_df, baseline_predictions, deep_predictions = evaluate_models(
        baseline_model=baseline_model,
        deep_model=deep_model,
        X_test_text=X_test,
        X_test_padded=X_test_padded,
        y_test_labels=y_test,
        y_test_encoded=y_test_encoded,
        label_encoder=label_encoder,
    )

    print_section("Visualizations")
    plot_class_distribution(processed_df)
    plot_training_curves(history)
    plot_confusion_matrix(y_test, deep_predictions)

    print_section("Pipeline Finished")
    print("Deep learning news classification project completed successfully.")


if __name__ == "__main__":
    main()