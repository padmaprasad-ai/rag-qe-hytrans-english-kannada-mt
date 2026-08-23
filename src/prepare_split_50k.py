# src/prepare_split_50k.py

import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_PARALLEL_FILE_50K,
    CLEAN_PARALLEL_FILE_50K,
    TRAIN_FILE_50K,
    VALID_FILE_50K,
    TEST_FILE_50K,
    PREPARE_50K_LOG,
    CHUNK_SIZE_50K,
    TRAIN_RATIO_50K,
    VALID_RATIO_50K,
    TEST_RATIO_50K,
    MIN_SOURCE_CHARS_50K,
    MIN_TARGET_CHARS_50K,
    MAX_SOURCE_CHARS_50K,
    MAX_TARGET_CHARS_50K,
    MIN_SOURCE_WORDS_50K,
    MIN_TARGET_WORDS_50K,
    MAX_SOURCE_WORDS_50K,
    MAX_TARGET_WORDS_50K,
    MAX_LENGTH_RATIO_50K,
    KANNADA_SCRIPT_REGEX_50K,
    ENABLE_KANNADA_SCRIPT_FILTER_50K,
    RANDOM_SEED,
    create_directories,
)


class PrepareSplit50K:
    """
    Cleans and splits 50K English-Kannada parallel corpus.

    Input:
        data/raw/parallel_50k.csv

    Outputs:
        data/processed/parallel_50k_clean.csv
        data/processed/train_50k.csv
        data/processed/valid_50k.csv
        data/processed/test_50k.csv
        outputs/prepare_50k_log.json
    """

    def __init__(self):
        self.total_read = 0
        self.total_after_basic_clean = 0
        self.total_after_advanced_clean = 0

        self.removed_empty = 0
        self.removed_char_length = 0
        self.removed_word_length = 0
        self.removed_length_ratio = 0
        self.removed_identical = 0
        self.removed_script = 0
        self.removed_duplicates = 0

    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""

        text = str(text).strip()
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def word_count(text):
        if not isinstance(text, str):
            return 0
        return len(text.split())

    @staticmethod
    def length_ratio(source_text, target_text):
        source_len = max(1, len(str(source_text).strip()))
        target_len = max(1, len(str(target_text).strip()))
        return max(source_len, target_len) / min(source_len, target_len)

    @staticmethod
    def has_kannada_script(text):
        return bool(re.search(KANNADA_SCRIPT_REGEX_50K, str(text)))

    def load_raw_file(self):
        if not RAW_PARALLEL_FILE_50K.exists():
            raise FileNotFoundError(
                f"50K raw file not found:\n{RAW_PARALLEL_FILE_50K}\n\n"
                "Please run Step 25 first: run_hf_50k_download.py"
            )

        df = pd.read_csv(RAW_PARALLEL_FILE_50K, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "domain",
            "source_lang",
            "target_lang",
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        return df[required_columns].copy()

    def clean_and_filter(self, df):
        print("Applying 50K cleaning and filtering...")

        self.total_read = len(df)

        df["source_text"] = df["source_text"].apply(self.clean_text)
        df["target_text"] = df["target_text"].apply(self.clean_text)

        before = len(df)
        df = df.dropna(subset=["source_text", "target_text"])
        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]
        self.removed_empty = before - len(df)

        before = len(df)
        df = df[df["source_text"].str.len() >= MIN_SOURCE_CHARS_50K]
        df = df[df["target_text"].str.len() >= MIN_TARGET_CHARS_50K]
        df = df[df["source_text"].str.len() <= MAX_SOURCE_CHARS_50K]
        df = df[df["target_text"].str.len() <= MAX_TARGET_CHARS_50K]
        self.removed_char_length = before - len(df)

        self.total_after_basic_clean = len(df)

        before = len(df)
        df = df[
            df["source_text"].str.lower().str.strip()
            != df["target_text"].str.lower().str.strip()
        ]
        self.removed_identical = before - len(df)

        df["source_word_count"] = df["source_text"].apply(self.word_count)
        df["target_word_count"] = df["target_text"].apply(self.word_count)

        before = len(df)
        df = df[df["source_word_count"] >= MIN_SOURCE_WORDS_50K]
        df = df[df["target_word_count"] >= MIN_TARGET_WORDS_50K]
        df = df[df["source_word_count"] <= MAX_SOURCE_WORDS_50K]
        df = df[df["target_word_count"] <= MAX_TARGET_WORDS_50K]
        self.removed_word_length = before - len(df)

        df["length_ratio"] = df.apply(
            lambda row: self.length_ratio(row["source_text"], row["target_text"]),
            axis=1,
        )

        before = len(df)
        df = df[df["length_ratio"] <= MAX_LENGTH_RATIO_50K]
        self.removed_length_ratio = before - len(df)

        if ENABLE_KANNADA_SCRIPT_FILTER_50K:
            before = len(df)
            df = df[df["target_text"].apply(self.has_kannada_script)]
            self.removed_script = before - len(df)

        before = len(df)
        df = df.drop_duplicates(subset=["source_text", "target_text"])
        self.removed_duplicates = before - len(df)

        df = df.drop(
            columns=[
                "source_word_count",
                "target_word_count",
                "length_ratio",
            ],
            errors="ignore",
        )

        df = df.reset_index(drop=True)

        self.total_after_advanced_clean = len(df)

        return df

    def split_data(self, df):
        if len(df) < 1000:
            raise ValueError(
                f"Only {len(df)} clean rows available. "
                "50K experiment requires a larger clean corpus."
            )

        train_df, temp_df = train_test_split(
            df,
            train_size=TRAIN_RATIO_50K,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        relative_valid_ratio = VALID_RATIO_50K / (VALID_RATIO_50K + TEST_RATIO_50K)

        valid_df, test_df = train_test_split(
            temp_df,
            train_size=relative_valid_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        return (
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def save_outputs(self, clean_df, train_df, valid_df, test_df):
        CLEAN_PARALLEL_FILE_50K.parent.mkdir(parents=True, exist_ok=True)
        PREPARE_50K_LOG.parent.mkdir(parents=True, exist_ok=True)

        clean_df.to_csv(CLEAN_PARALLEL_FILE_50K, index=False, encoding="utf-8-sig")
        train_df.to_csv(TRAIN_FILE_50K, index=False, encoding="utf-8-sig")
        valid_df.to_csv(VALID_FILE_50K, index=False, encoding="utf-8-sig")
        test_df.to_csv(TEST_FILE_50K, index=False, encoding="utf-8-sig")

        log_data = {
            "input_file": str(RAW_PARALLEL_FILE_50K),
            "clean_file": str(CLEAN_PARALLEL_FILE_50K),
            "train_file": str(TRAIN_FILE_50K),
            "valid_file": str(VALID_FILE_50K),
            "test_file": str(TEST_FILE_50K),
            "counts": {
                "raw_rows": self.total_read,
                "after_basic_clean": self.total_after_basic_clean,
                "after_advanced_clean": self.total_after_advanced_clean,
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "test_rows": len(test_df),
            },
            "removed_counts": {
                "empty_removed": self.removed_empty,
                "char_length_removed": self.removed_char_length,
                "identical_source_target_removed": self.removed_identical,
                "word_length_removed": self.removed_word_length,
                "length_ratio_removed": self.removed_length_ratio,
                "script_filter_removed": self.removed_script,
                "duplicates_removed": self.removed_duplicates,
            },
            "split_ratios": {
                "train": TRAIN_RATIO_50K,
                "valid": VALID_RATIO_50K,
                "test": TEST_RATIO_50K,
            },
        }

        with open(PREPARE_50K_LOG, "w", encoding="utf-8") as file:
            json.dump(log_data, file, indent=4, ensure_ascii=False)

    def run(self):
        create_directories()

        print("Step 26: 50K corpus cleaning and split started.")
        print(f"Input file: {RAW_PARALLEL_FILE_50K}")

        df = self.load_raw_file()

        print(f"Raw rows loaded: {len(df)}")

        clean_df = self.clean_and_filter(df)

        print(f"Rows after cleaning: {len(clean_df)}")

        train_df, valid_df, test_df = self.split_data(clean_df)

        self.save_outputs(clean_df, train_df, valid_df, test_df)

        print("\n50K cleaning and split completed successfully.")

        print("\nCleaning summary:")
        print(f"Raw rows                       : {self.total_read}")
        print(f"Rows after basic cleaning      : {self.total_after_basic_clean}")
        print(f"Rows after advanced cleaning   : {self.total_after_advanced_clean}")
        print(f"Empty rows removed             : {self.removed_empty}")
        print(f"Character-length removed       : {self.removed_char_length}")
        print(f"Identical source-target removed: {self.removed_identical}")
        print(f"Word-length removed            : {self.removed_word_length}")
        print(f"Length-ratio removed           : {self.removed_length_ratio}")
        print(f"Script-filter removed          : {self.removed_script}")
        print(f"Duplicate pairs removed        : {self.removed_duplicates}")

        print("\nSplit summary:")
        print(f"Train rows     : {len(train_df)}")
        print(f"Validation rows: {len(valid_df)}")
        print(f"Test rows      : {len(test_df)}")

        print("\nFiles saved:")
        print(f"Clean corpus: {CLEAN_PARALLEL_FILE_50K}")
        print(f"Train file  : {TRAIN_FILE_50K}")
        print(f"Valid file  : {VALID_FILE_50K}")
        print(f"Test file   : {TEST_FILE_50K}")
        print(f"Log file    : {PREPARE_50K_LOG}")

        print("\nSample training rows:")
        print(train_df.head(5).to_string(index=False))


def run_prepare_split_50k():
    processor = PrepareSplit50K()
    processor.run()


if __name__ == "__main__":
    run_prepare_split_50k()