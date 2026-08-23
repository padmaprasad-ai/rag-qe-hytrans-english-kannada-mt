# src/large_data_prepare.py

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.config import (
    LARGE_RAW_PARALLEL_FILE,
    LARGE_CLEAN_PARALLEL_FILE,
    LARGE_DATA_PREPARATION_LOG,
    LARGE_DATA_CHUNK_SIZE,
    LARGE_MIN_SOURCE_CHARS,
    LARGE_MIN_TARGET_CHARS,
    LARGE_MAX_SOURCE_CHARS,
    LARGE_MAX_TARGET_CHARS,
    LARGE_MAX_LENGTH_RATIO,
    create_directories,
)


class LargeParallelCorpusPreparer:
    """
    Large parallel corpus preparation module for low-resource MT.

    This module supports large CSV/TSV files and converts them into the standard format:

        source_text,target_text,domain,source_lang,target_lang

    It performs:
        - flexible column detection
        - text cleaning
        - empty row removal
        - duplicate removal
        - basic length filtering
        - chunk-wise processing for large files
    """

    def __init__(
        self,
        input_file: Path = LARGE_RAW_PARALLEL_FILE,
        output_file: Path = LARGE_CLEAN_PARALLEL_FILE,
        chunk_size: int = LARGE_DATA_CHUNK_SIZE,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size

        self.total_rows_read = 0
        self.total_rows_written = 0
        self.total_empty_removed = 0
        self.total_length_filtered = 0
        self.total_duplicates_removed = 0

        self.seen_pairs = set()

    @staticmethod
    def normalize_column_name(column_name: str) -> str:
        column_name = str(column_name).strip().lower()
        column_name = column_name.replace(" ", "_")
        column_name = column_name.replace("-", "_")
        column_name = column_name.replace(".", "_")
        column_name = column_name.replace("/", "_")
        return column_name

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans text for machine translation training.
        Keeps punctuation and script-specific characters.
        """
        if pd.isna(text):
            return ""

        text = str(text).strip()

        # Remove zero-width characters often found in Indic text
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove HTML-like tags if present
        text = re.sub(r"<[^>]+>", "", text)

        return text.strip()

    def detect_separator(self) -> str:
        """
        Detects delimiter based on file extension and first line.
        """
        suffix = self.input_file.suffix.lower()

        if suffix == ".tsv":
            return "\t"

        if suffix == ".csv":
            return ","

        # For .txt or unknown files, inspect first line
        with open(self.input_file, "r", encoding="utf-8", errors="ignore") as file:
            first_line = file.readline()

        if "\t" in first_line:
            return "\t"

        return ","

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts different source/target column names into source_text and target_text.
        """
        original_columns = list(df.columns)
        df.columns = [self.normalize_column_name(col) for col in df.columns]

        source_candidates = [
            "source_text",
            "source",
            "src",
            "input",
            "english",
            "en",
            "text_en",
            "sentence_en",
            "source_sentence",
            "sentence_source",
            "sentence1",
            "src_text",
        ]

        target_candidates = [
            "target_text",
            "target",
            "tgt",
            "output",
            "translation",
            "kannada",
            "kn",
            "text_kn",
            "sentence_kn",
            "target_sentence",
            "sentence_target",
            "sentence2",
            "tgt_text",
        ]

        column_mapping = {}

        for col in df.columns:
            if col in source_candidates:
                column_mapping[col] = "source_text"

            if col in target_candidates:
                column_mapping[col] = "target_text"

        df = df.rename(columns=column_mapping)

        if "source_text" not in df.columns or "target_text" not in df.columns:
            raise ValueError(
                "\nCould not detect source and target columns.\n\n"
                f"Original columns: {original_columns}\n"
                f"Normalized columns: {list(df.columns)}\n\n"
                "Please rename your file columns to one of these formats:\n"
                "source_text,target_text\n"
                "english,kannada\n"
                "source,target\n"
                "src,tgt\n"
            )

        return df

    def add_metadata_columns(
        self,
        df: pd.DataFrame,
        domain: str = "general",
        source_lang: str = "en",
        target_lang: str = "kn",
    ) -> pd.DataFrame:
        if "domain" not in df.columns:
            df["domain"] = domain

        if "source_lang" not in df.columns:
            df["source_lang"] = source_lang

        if "target_lang" not in df.columns:
            df["target_lang"] = target_lang

        return df

    def apply_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        before_empty_filter = len(df)

        df["source_text"] = df["source_text"].apply(self.clean_text)
        df["target_text"] = df["target_text"].apply(self.clean_text)

        df = df.dropna(subset=["source_text", "target_text"])
        df = df[df["source_text"].str.len() >= LARGE_MIN_SOURCE_CHARS]
        df = df[df["target_text"].str.len() >= LARGE_MIN_TARGET_CHARS]

        self.total_empty_removed += before_empty_filter - len(df)

        before_length_filter = len(df)

        df["source_len"] = df["source_text"].str.len()
        df["target_len"] = df["target_text"].str.len()

        df = df[df["source_len"] <= LARGE_MAX_SOURCE_CHARS]
        df = df[df["target_len"] <= LARGE_MAX_TARGET_CHARS]

        df["length_ratio"] = df[["source_len", "target_len"]].max(axis=1) / df[
            ["source_len", "target_len"]
        ].min(axis=1)

        df = df[df["length_ratio"] <= LARGE_MAX_LENGTH_RATIO]

        self.total_length_filtered += before_length_filter - len(df)

        df = df.drop(columns=["source_len", "target_len", "length_ratio"])

        return df

    def remove_cross_chunk_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate sentence pairs across chunks.
        """
        unique_rows = []

        before_duplicate_filter = len(df)

        for _, row in df.iterrows():
            pair_key = (
                str(row["source_text"]).strip().lower(),
                str(row["target_text"]).strip().lower(),
            )

            if pair_key not in self.seen_pairs:
                self.seen_pairs.add(pair_key)
                unique_rows.append(row)

        unique_df = pd.DataFrame(unique_rows)

        self.total_duplicates_removed += before_duplicate_filter - len(unique_df)

        return unique_df

    def process_chunk(
        self,
        chunk_df: pd.DataFrame,
        domain: str,
        source_lang: str,
        target_lang: str,
    ) -> pd.DataFrame:
        chunk_df = self.standardize_columns(chunk_df)

        chunk_df = chunk_df[["source_text", "target_text"]].copy()

        chunk_df = self.apply_basic_filters(chunk_df)

        chunk_df = self.remove_cross_chunk_duplicates(chunk_df)

        if len(chunk_df) == 0:
            return chunk_df

        chunk_df = self.add_metadata_columns(
            chunk_df,
            domain=domain,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        chunk_df = chunk_df[
            [
                "source_text",
                "target_text",
                "domain",
                "source_lang",
                "target_lang",
            ]
        ]

        return chunk_df

    def prepare_large_corpus(
        self,
        domain: str = "general",
        source_lang: str = "en",
        target_lang: str = "kn",
    ):
        create_directories()

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"\nLarge parallel file not found:\n{self.input_file}\n\n"
                "Please place your large parallel corpus at:\n"
                "data/raw/large_parallel.csv\n"
            )

        separator = self.detect_separator()

        print("Large parallel corpus preparation started.")
        print(f"Input file : {self.input_file}")
        print(f"Output file: {self.output_file}")
        print(f"Separator  : {'TAB' if separator == chr(9) else separator}")
        print(f"Chunk size : {self.chunk_size}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if self.output_file.exists():
            self.output_file.unlink()

        first_write = True
        chunk_number = 0

        reader = pd.read_csv(
            self.input_file,
            sep=separator,
            chunksize=self.chunk_size,
            encoding="utf-8-sig",
            on_bad_lines="skip",
            low_memory=False,
        )

        for chunk_df in reader:
            chunk_number += 1
            self.total_rows_read += len(chunk_df)

            print(f"\nProcessing chunk {chunk_number} | Rows read: {len(chunk_df)}")

            cleaned_chunk = self.process_chunk(
                chunk_df=chunk_df,
                domain=domain,
                source_lang=source_lang,
                target_lang=target_lang,
            )

            if len(cleaned_chunk) > 0:
                cleaned_chunk.to_csv(
                    self.output_file,
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False,
                    encoding="utf-8-sig",
                )

                first_write = False
                self.total_rows_written += len(cleaned_chunk)

            print(f"Cleaned rows written from this chunk: {len(cleaned_chunk)}")

        self.save_log(
            domain=domain,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        print("\nLarge corpus preparation completed successfully.")
        print(f"Total rows read              : {self.total_rows_read}")
        print(f"Total clean rows written     : {self.total_rows_written}")
        print(f"Empty/short rows removed     : {self.total_empty_removed}")
        print(f"Length-filtered rows removed : {self.total_length_filtered}")
        print(f"Duplicate pairs removed      : {self.total_duplicates_removed}")
        print(f"Clean corpus saved at        : {self.output_file}")
        print(f"Preparation log saved at     : {LARGE_DATA_PREPARATION_LOG}")

    def save_log(
        self,
        domain: str,
        source_lang: str,
        target_lang: str,
    ):
        log_data = {
            "input_file": str(self.input_file),
            "output_file": str(self.output_file),
            "domain": domain,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "chunk_size": self.chunk_size,
            "total_rows_read": self.total_rows_read,
            "total_clean_rows_written": self.total_rows_written,
            "empty_or_short_rows_removed": self.total_empty_removed,
            "length_filtered_rows_removed": self.total_length_filtered,
            "duplicate_pairs_removed": self.total_duplicates_removed,
            "filters": {
                "min_source_chars": LARGE_MIN_SOURCE_CHARS,
                "min_target_chars": LARGE_MIN_TARGET_CHARS,
                "max_source_chars": LARGE_MAX_SOURCE_CHARS,
                "max_target_chars": LARGE_MAX_TARGET_CHARS,
                "max_length_ratio": LARGE_MAX_LENGTH_RATIO,
            },
        }

        LARGE_DATA_PREPARATION_LOG.parent.mkdir(parents=True, exist_ok=True)

        with open(LARGE_DATA_PREPARATION_LOG, "w", encoding="utf-8") as file:
            json.dump(log_data, file, indent=4, ensure_ascii=False)


def run_large_data_preparation():
    preparer = LargeParallelCorpusPreparer()

    preparer.prepare_large_corpus(
        domain="general",
        source_lang="en",
        target_lang="kn",
    )


if __name__ == "__main__":
    run_large_data_preparation()