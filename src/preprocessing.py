import re

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    LABEL_ENCODER_FILE,
    MAX_SEQUENCE_LENGTH,
    MAX_WORDS,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TOKENIZER_FILE,
    VALIDATION_SIZE,
)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = df.copy()
    processed_df["clean_text"] = processed_df["text"].apply(clean_text)
    processed_df = processed_df[processed_df["clean_text"].str.len() > 10]
    # Avoid an identical cleaned document appearing in different splits.
    processed_df = processed_df.drop_duplicates(subset=["clean_text"])
    return processed_df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    """Return train, validation, test sets with disjoint original row indices.

    TEST_SIZE is a fraction of the whole dataset. VALIDATION_SIZE is a
    fraction of the remaining training pool, not the full dataset.
    """
    if df.empty:
        raise ValueError("No rows are available after preprocessing.")
    if df["category"].nunique() < 2:
        raise ValueError("At least two classes are required.")

    X = df["clean_text"]
    y = df["category"]

    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train_val,
        )
    except ValueError as error:
        raise ValueError(
            "Not enough examples per class for independent stratified "
            "train/validation/test splits. Use the full 20 Newsgroups dataset "
            "instead of the tiny eight-row demo."
        ) from error

    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_labels(y_train, y_val, y_test):
    """Fit label mapping on training labels only."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder


def tokenize_text(X_train, X_val, X_test):
    """Fit tokenizer on training texts only; transform val and test."""
    # Lazy import lets split tests run without loading TensorFlow.
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def transform(texts):
        return pad_sequences(
            tokenizer.texts_to_sequences(texts),
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )

    X_train_padded = transform(X_train)
    X_val_padded = transform(X_val)
    X_test_padded = transform(X_test)
    joblib.dump(tokenizer, TOKENIZER_FILE)
    return X_train_padded, X_val_padded, X_test_padded, tokenizer
