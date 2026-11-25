import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from src.config import NEWS_CATEGORIES, PROCESSED_DATA_FILE, PROCESSED_DATA_DIR, RANDOM_STATE


def load_news_dataset() -> pd.DataFrame:
    """Load the four selected newsgroup classes, failing clearly if unavailable.

    Do not silently train/evaluate on a tiny synthetic substitute.
    """
    try:
        dataset = fetch_20newsgroups(
            subset="all",
            categories=NEWS_CATEGORIES,
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    except Exception as error:
        raise RuntimeError(
            "The 20 Newsgroups dataset is unavailable. Connect to the "
            "internet once to download/cache it (then rerun python main.py). "
            "This pipeline intentionally does not report model metrics "
            "from an eight-row fallback demo dataset."
        ) from error

    df = pd.DataFrame(
        {
            "text": dataset.data,
            "category": [
                dataset.target_names[target] for target in dataset.target
            ],
        }
    )
    df = df.dropna().copy()
    df["text"] = df["text"].astype(str)
    df["category"] = df["category"].astype(str)
    return df.loc[df["text"].str.strip().str.len() > 0].reset_index(drop=True)


def save_processed_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Processed dataset saved to: {PROCESSED_DATA_FILE}")


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\nDataset shape:")
    print(df.shape)
    print("\nClass distribution:")
    print(df["category"].value_counts())
