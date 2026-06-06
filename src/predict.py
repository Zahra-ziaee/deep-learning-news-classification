import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import DEEP_MODEL_FILE, LABEL_ENCODER_FILE, MAX_SEQUENCE_LENGTH, TOKENIZER_FILE
from src.preprocessing import clean_text


def load_prediction_artifacts():
    model = load_model(DEEP_MODEL_FILE)
    tokenizer = joblib.load(TOKENIZER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    return model, tokenizer, label_encoder


def predict_news_category(text: str) -> dict:
    model, tokenizer, label_encoder = load_prediction_artifacts()

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded_sequence)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_category = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])

    class_probabilities = {
        label_encoder.inverse_transform([index])[0]: round(float(probability), 4)
        for index, probability in enumerate(probabilities)
    }

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{confidence * 100:.2f}%",
        "class_probabilities": class_probabilities,
    }