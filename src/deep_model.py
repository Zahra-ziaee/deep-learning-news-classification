from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential

from src.config import (
    BATCH_SIZE,
    DEEP_MODEL_FILE,
    EMBEDDING_DIM,
    EPOCHS,
    MAX_SEQUENCE_LENGTH,
    MAX_WORDS,
    MODELS_DIR,
)


def build_bilstm_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Embedding(
                input_dim=MAX_WORDS,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
            ),
            Bidirectional(LSTM(64)),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_deep_model(X_train_padded, y_train_encoded, X_test_padded, y_test_encoded, num_classes):
    print("\nTraining Deep Learning BiLSTM model...")

    model = build_bilstm_model(num_classes=num_classes)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_padded,
        y_train_encoded,
        validation_data=(X_test_padded, y_test_encoded),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(DEEP_MODEL_FILE)

    print(f"Deep learning model saved to: {DEEP_MODEL_FILE}")

    return model, history