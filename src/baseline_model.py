import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import BASELINE_MODEL_FILE, MODELS_DIR, RANDOM_STATE


def build_baseline_model() -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def train_baseline_model(X_train, y_train):
    print("\nTraining TF-IDF + Logistic Regression baseline model...")

    model = build_baseline_model()
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASELINE_MODEL_FILE)

    print(f"Baseline model saved to: {BASELINE_MODEL_FILE}")

    return model