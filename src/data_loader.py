import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from src.config import NEWS_CATEGORIES, PROCESSED_DATA_FILE, PROCESSED_DATA_DIR, RANDOM_STATE


def load_news_dataset() -> pd.DataFrame:
    """
    Loads a selected subset of the 20 Newsgroups dataset.
    If internet download fails, it falls back to a small local demo dataset.
    """

    try:
        dataset = fetch_20newsgroups(
            subset="all",
            categories=NEWS_CATEGORIES,
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        df = pd.DataFrame(
            {
                "text": dataset.data,
                "category": [dataset.target_names[target] for target in dataset.target],
            }
        )

    except Exception as error:
        print("Could not download 20 Newsgroups dataset.")
        print(f"Reason: {error}")
        print("Using fallback demo dataset instead.")

        fallback_data = [
            ("The new graphics card improves rendering performance and image processing.", "comp.graphics"),
            ("3D modeling software uses rendering engines and computer graphics algorithms.", "comp.graphics"),
            ("The baseball team won the game after a strong pitching performance.", "rec.sport.baseball"),
            ("The player hit a home run in the final inning of the baseball match.", "rec.sport.baseball"),
            ("Doctors discussed symptoms, diagnosis, treatment, and patient health outcomes.", "sci.med"),
            ("Medical research shows progress in disease prevention and clinical treatment.", "sci.med"),
            ("The political conflict affected regional negotiations and international relations.", "talk.politics.mideast"),
            ("Diplomatic talks focused on peace, security, and Middle East policy.", "talk.politics.mideast"),
        ]

        df = pd.DataFrame(fallback_data, columns=["text", "category"])

    df = df.dropna()
    df["text"] = df["text"].astype(str)
    df["category"] = df["category"].astype(str)

    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

    return df


def save_processed_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Processed dataset saved to: {PROCESSED_DATA_FILE}")


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\nDataset shape:")
    print(df.shape)

    print("\nClass distribution:")
    print(df["category"].value_counts())