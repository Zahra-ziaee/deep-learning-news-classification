from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "news_data.csv"

BASELINE_MODEL_FILE = MODELS_DIR / "baseline_tfidf_logistic_model.joblib"
DEEP_MODEL_FILE = MODELS_DIR / "news_bilstm_model.keras"
TOKENIZER_FILE = MODELS_DIR / "tokenizer.joblib"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.joblib"

METRICS_FILE = RESULTS_DIR / "metrics.csv"
CLASSIFICATION_REPORT_FILE = RESULTS_DIR / "classification_report.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
EPOCHS = 4
BATCH_SIZE = 32

NEWS_CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.mideast",
]