# src/large_data_split.py

import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LARGE_CLEAN_PARALLEL_FILE,
    LARGE_FILTERED_PARALLEL_FILE,
    TRAIN_LARGE_FILE,
    VALID_LARGE_FILE,
    TEST_LARGE_FILE,
    LARGE_SPLIT_LOG,
    LARGE_TRAIN_RATIO,
    LARGE_VALID_RATIO,
    LARGE_TEST_RATIO,
    LARGE_MIN_SOURCE_WORDS,
    LARGE_MIN_TARGET_WORDS,
    LARGE_MAX_SOURCE_WORDS,
    LARGE_MAX_TARGET_WORDS,
    LARGE_ADVANCED_MAX_LENGTH_RATIO,
    KANNADA_SCRIPT_REGEX,
    ENABLE_KANNADA_SCRIPT_FILTER,
    RANDOM_SEED,
    create_directories,
)


class LargeCorpusSplitter:
    """
    Advanced filtering and train/validation/test splitting module
    for large parallel machine translation corpora.

    Input:
        data/processed/large_parallel_clean.csv

    Outputs:
        data/processed/large_parallel_filtered.csv
        data/processed/train_large.csv
        data/processed/valid_large.csv
        data/processed/test_large.csv
        outputs/large_split_log.json
    """

    def __init__(self):
        self.original_count = 0
        self.after_filter_count = 0

        self.removed_empty = 0
        self.removed_identical = 0
        self.removed_word_length = 0
        self.removed_length_ratio = 0
        self.removed_script = 0
        self.removed_duplicates = 0

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).strip()
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def word_count(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return len(text.strip().split())

    @staticmethod
    def safe_length_ratio(source_text: str, target_text: str) -> float:
        source_len = max(1, len(str(source_text).strip()))
        target_len = max(1, len(str(target_text).strip()))
        return max(source_len, target_len) / min(source_len, target_len)

    @staticmethod
    def has_kannada_script(text: str) -> bool:
        if not isinstance(text, str):
            return False
        return bool(re.search(KANNADA_SCRIPT_REGEX, text))

    def load_clean_corpus(self) -> pd.DataFrame:
        if not LARGE_CLEAN_PARALLEL_FILE.exists():
            raise FileNotFoundError(
                f"Clean large corpus not found:\n{LARGE_CLEAN_PARALLEL_FILE}\n\n"
                "Please run Step 11 first: run_large_data_prepare.py"
            )

        df = pd.read_csv(LARGE_CLEAN_PARALLEL_FILE, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "domain",
            "source_lang",
            "target_lang",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Detected columns: {list(df.columns)}"
            )

        self.original_count = len(df)

        return df

    def apply_advanced_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Applying advanced MT quality filters...")

        df = df.copy()

        df["source_text"] = df["source_text"].apply(self.clean_text)
        df["target_text"] = df["target_text"].apply(self.clean_text)

        before = len(df)
        df = df.dropna(subset=["source_text", "target_text"])
        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]
        self.removed_empty = before - len(df)

        before = len(df)
        df = df[
            df["source_text"].str.lower().str.strip()
            != df["target_text"].str.lower().str.strip()
        ]
        self.removed_identical = before - len(df)

        df["source_word_count"] = df["source_text"].apply(self.word_count)
        df["target_word_count"] = df["target_text"].apply(self.word_count)

        before = len(df)
        df = df[df["source_word_count"] >= LARGE_MIN_SOURCE_WORDS]
        df = df[df["target_word_count"] >= LARGE_MIN_TARGET_WORDS]
        df = df[df["source_word_count"] <= LARGE_MAX_SOURCE_WORDS]
        df = df[df["target_word_count"] <= LARGE_MAX_TARGET_WORDS]
        self.removed_word_length = before - len(df)

        df["length_ratio"] = df.apply(
            lambda row: self.safe_length_ratio(
                row["source_text"],
                row["target_text"],
            ),
            axis=1,
        )

        before = len(df)
        df = df[df["length_ratio"] <= LARGE_ADVANCED_MAX_LENGTH_RATIO]
        self.removed_length_ratio = before - len(df)

        if ENABLE_KANNADA_SCRIPT_FILTER:
            before = len(df)
            df = df[df["target_text"].apply(self.has_kannada_script)]
            self.removed_script = before - len(df)
        else:
            self.removed_script = 0

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

        self.after_filter_count = len(df)

        return df

    @staticmethod
    def can_stratify(df: pd.DataFrame, column: str = "domain") -> bool:
        if column not in df.columns:
            return False

        value_counts = df[column].value_counts()

        if len(value_counts) < 2:
            return False

        if value_counts.min() < 2:
            return False

        return True

    def split_dataset(self, df: pd.DataFrame):
        if len(df) < 100:
            raise ValueError(
                f"Only {len(df)} rows available after filtering. "
                "For large corpus experiments, keep at least 100 rows."
            )

        stratify_column = df["domain"] if self.can_stratify(df, "domain") else None

        train_df, temp_df = train_test_split(
            df,
            train_size=LARGE_TRAIN_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=stratify_column,
        )

        relative_valid_ratio = LARGE_VALID_RATIO / (LARGE_VALID_RATIO + LARGE_TEST_RATIO)

        stratify_temp = temp_df["domain"] if self.can_stratify(temp_df, "domain") else None

        valid_df, test_df = train_test_split(
            temp_df,
            train_size=relative_valid_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=stratify_temp,
        )

        return (
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    @staticmethod
    def domain_distribution(df: pd.DataFrame) -> dict:
        if "domain" not in df.columns:
            return {}
        return df["domain"].value_counts().to_dict()

    def save_outputs(
        self,
        filtered_df: pd.DataFrame,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        filtered_df.to_csv(
            LARGE_FILTERED_PARALLEL_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        train_df.to_csv(TRAIN_LARGE_FILE, index=False, encoding="utf-8-sig")
        valid_df.to_csv(VALID_LARGE_FILE, index=False, encoding="utf-8-sig")
        test_df.to_csv(TEST_LARGE_FILE, index=False, encoding="utf-8-sig")

        log_data = {
            "input_file": str(LARGE_CLEAN_PARALLEL_FILE),
            "filtered_file": str(LARGE_FILTERED_PARALLEL_FILE),
            "train_file": str(TRAIN_LARGE_FILE),
            "valid_file": str(VALID_LARGE_FILE),
            "test_file": str(TEST_LARGE_FILE),
            "original_count": self.original_count,
            "after_filter_count": self.after_filter_count,
            "removed_counts": {
                "empty_removed": self.removed_empty,
                "identical_source_target_removed": self.removed_identical,
                "word_length_removed": self.removed_word_length,
                "length_ratio_removed": self.removed_length_ratio,
                "script_filter_removed": self.removed_script,
                "duplicates_removed": self.removed_duplicates,
            },
            "split_counts": {
                "train": len(train_df),
                "valid": len(valid_df),
                "test": len(test_df),
            },
            "split_ratios": {
                "train": LARGE_TRAIN_RATIO,
                "valid": LARGE_VALID_RATIO,
                "test": LARGE_TEST_RATIO,
            },
            "domain_distribution": {
                "full_filtered": self.domain_distribution(filtered_df),
                "train": self.domain_distribution(train_df),
                "valid": self.domain_distribution(valid_df),
                "test": self.domain_distribution(test_df),
            },
            "filters": {
                "min_source_words": LARGE_MIN_SOURCE_WORDS,
                "min_target_words": LARGE_MIN_TARGET_WORDS,
                "max_source_words": LARGE_MAX_SOURCE_WORDS,
                "max_target_words": LARGE_MAX_TARGET_WORDS,
                "max_length_ratio": LARGE_ADVANCED_MAX_LENGTH_RATIO,
                "kannada_script_filter_enabled": ENABLE_KANNADA_SCRIPT_FILTER,
            },
        }

        LARGE_SPLIT_LOG.parent.mkdir(parents=True, exist_ok=True)

        with open(LARGE_SPLIT_LOG, "w", encoding="utf-8") as file:
            json.dump(log_data, file, indent=4, ensure_ascii=False)

    def run(self):
        create_directories()

        print("Step 12: Large corpus advanced filtering and split started.")

        df = self.load_clean_corpus()

        print(f"Original clean corpus rows: {len(df)}")

        filtered_df = self.apply_advanced_filters(df)

        print(f"Rows after advanced filtering: {len(filtered_df)}")

        train_df, valid_df, test_df = self.split_dataset(filtered_df)

        self.save_outputs(filtered_df, train_df, valid_df, test_df)

        print("\nStep 12 completed successfully.")

        print("\nFiltering summary:")
        print(f"Original rows                    : {self.original_count}")
        print(f"Rows after filtering             : {self.after_filter_count}")
        print(f"Empty rows removed               : {self.removed_empty}")
        print(f"Identical source-target removed  : {self.removed_identical}")
        print(f"Word-length filtered rows removed: {self.removed_word_length}")
        print(f"Length-ratio filtered rows removed: {self.removed_length_ratio}")
        print(f"Script-filtered rows removed     : {self.removed_script}")
        print(f"Duplicate rows removed           : {self.removed_duplicates}")

        print("\nSplit summary:")
        print(f"Train rows     : {len(train_df)}")
        print(f"Validation rows: {len(valid_df)}")
        print(f"Test rows      : {len(test_df)}")

        print("\nFiles saved:")
        print(f"Filtered corpus: {LARGE_FILTERED_PARALLEL_FILE}")
        print(f"Train file     : {TRAIN_LARGE_FILE}")
        print(f"Validation file: {VALID_LARGE_FILE}")
        print(f"Test file      : {TEST_LARGE_FILE}")
        print(f"Split log      : {LARGE_SPLIT_LOG}")

        print("\nSample training rows:")
        print(train_df.head(5).to_string(index=False))


def run_large_data_split():
    splitter = LargeCorpusSplitter()
    splitter.run()


if __name__ == "__main__":
    run_large_data_split()