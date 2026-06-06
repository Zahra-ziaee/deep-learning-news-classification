import re
from typing import Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.config import (
    LABEL_ENCODER_FILE,
    MAX_SEQUENCE_LENGTH,
    MAX_WORDS,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TOKENIZER_FILE,
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
    processed_df = processed_df.reset_index(drop=True)
    return processed_df


def split_dataset(df: pd.DataFrame):
    X = df["clean_text"]
    y = df["category"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def encode_labels(y_train, y_test):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    joblib.dump(label_encoder, LABEL_ENCODER_FILE)

    return y_train_encoded, y_test_encoded, label_encoder


def tokenize_text(X_train, X_test):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>",
    )

    tokenizer.fit_on_texts(X_train)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    joblib.dump(tokenizer, TOKENIZER_FILE)

    return X_train_padded, X_test_padded, tokenizer