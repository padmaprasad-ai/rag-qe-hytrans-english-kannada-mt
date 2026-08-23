# src/preprocess.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_PARALLEL_FILE,
    TRAIN_FILE,
    VALID_FILE,
    TEST_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    create_directories,
)


def clean_text(text: str) -> str:
    """
    Basic text cleaning for machine translation data.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    # Remove zero-width characters commonly found in Indic and web text
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    return text


def validate_columns(df: pd.DataFrame):
    required_columns = ["source_text", "target_text"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}. "
                f"Your CSV must contain source_text and target_text."
            )


def preprocess_dataset():
    create_directories()

    if not RAW_PARALLEL_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PARALLEL_FILE}\n"
            "Please place parallel.csv inside data/raw/"
        )

    df = pd.read_csv(RAW_PARALLEL_FILE)
    validate_columns(df)

    df["source_text"] = df["source_text"].apply(clean_text)
    df["target_text"] = df["target_text"].apply(clean_text)

    df = df.dropna(subset=["source_text", "target_text"])
    df = df[df["source_text"].str.len() > 0]
    df = df[df["target_text"].str.len() > 0]

    df = df.drop_duplicates(subset=["source_text", "target_text"])
    df = df.reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "Dataset is too small for train/validation/test split. "
            "Please add at least 10 sentence pairs for testing the pipeline."
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8")
    valid_df.to_csv(VALID_FILE, index=False, encoding="utf-8")
    test_df.to_csv(TEST_FILE, index=False, encoding="utf-8")

    print("Preprocessing completed successfully.")
    print(f"Total records: {len(df)}")
    print(f"Train records: {len(train_df)}")
    print(f"Validation records: {len(valid_df)}")
    print(f"Test records: {len(test_df)}")
    print(f"Train file saved at: {TRAIN_FILE}")
    print(f"Validation file saved at: {VALID_FILE}")
    print(f"Test file saved at: {TEST_FILE}")


if __name__ == "__main__":
    preprocess_dataset()