# src/download_parallel_from_hf.py

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from src.config import (
    HF_DATASET_NAME,
    HF_DATASET_CONFIG,
    HF_DATASET_SPLIT,
    HF_SOURCE_LANG,
    HF_TARGET_LANG,
    HF_MAX_SAMPLES,
    HF_OUTPUT_FILE,
    OUTPUT_DIR,
    create_directories,
)


class HuggingFaceParallelCorpusDownloader:
    """
    Downloads or streams a parallel corpus from Hugging Face
    and converts it into the project standard format:

        source_text,target_text,domain,source_lang,target_lang

    Default dataset:
        ai4bharat/samanantar, config='kn'

    Samanantar fields:
        src = English sentence
        tgt = Indic language sentence
        data_source = source/domain information
    """

    def __init__(
        self,
        dataset_name: str = HF_DATASET_NAME,
        dataset_config: str = HF_DATASET_CONFIG,
        split: str = HF_DATASET_SPLIT,
        source_lang: str = HF_SOURCE_LANG,
        target_lang: str = HF_TARGET_LANG,
        max_samples: int = HF_MAX_SAMPLES,
        output_file: Path = HF_OUTPUT_FILE,
        streaming: bool = True,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_samples = max_samples
        self.output_file = output_file
        self.streaming = streaming

        self.rows_seen = 0
        self.rows_written = 0
        self.rows_skipped = 0

    @staticmethod
    def clean_text(text: str) -> str:
        if text is None:
            return ""

        text = str(text).strip()
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")
        text = " ".join(text.split())

        return text

    def extract_pair(self, example: Dict) -> Optional[Tuple[str, str, str]]:
        """
        Supports multiple common Hugging Face parallel corpus formats.

        Format 1: Samanantar
            {'src': 'English sentence', 'tgt': 'Kannada sentence', 'data_source': '...'}

        Format 2: OPUS-style
            {'translation': {'en': '...', 'kn': '...'}}

        Format 3: Standard custom fields
            {'source_text': '...', 'target_text': '...'}
        """

        source_text = ""
        target_text = ""
        domain = "general"

        # Samanantar format
        if "src" in example and "tgt" in example:
            source_text = example.get("src", "")
            target_text = example.get("tgt", "")
            domain = example.get("data_source", "samanantar")

        # OPUS / translation-dict format
        elif "translation" in example and isinstance(example["translation"], dict):
            translation = example["translation"]
            source_text = translation.get(self.source_lang, "")
            target_text = translation.get(self.target_lang, "")
            domain = "opus"

        # Already standardized format
        elif "source_text" in example and "target_text" in example:
            source_text = example.get("source_text", "")
            target_text = example.get("target_text", "")
            domain = example.get("domain", "general")

        # English-Kannada direct format
        elif "english" in example and "kannada" in example:
            source_text = example.get("english", "")
            target_text = example.get("kannada", "")
            domain = "general"

        source_text = self.clean_text(source_text)
        target_text = self.clean_text(target_text)
        domain = self.clean_text(domain)

        if not source_text or not target_text:
            return None

        return source_text, target_text, domain

    def load_hf_dataset(self):
        print("Loading Hugging Face dataset...")
        print(f"Dataset : {self.dataset_name}")
        print(f"Config  : {self.dataset_config}")
        print(f"Split   : {self.split}")
        print(f"Streaming: {self.streaming}")

        try:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=True,
            )
        except TypeError:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=self.streaming,
            )

        return dataset

    def download_and_save(self):
        create_directories()

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        dataset = self.load_hf_dataset()

        fieldnames = [
            "source_text",
            "target_text",
            "domain",
            "source_lang",
            "target_lang",
        ]

        print(f"\nWriting output file:")
        print(self.output_file)
        print(f"\nMaximum samples to save: {self.max_samples}")

        with open(self.output_file, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            progress_bar = tqdm(total=self.max_samples)

            for example in dataset:
                self.rows_seen += 1

                extracted = self.extract_pair(example)

                if extracted is None:
                    self.rows_skipped += 1
                    continue

                source_text, target_text, domain = extracted

                writer.writerow(
                    {
                        "source_text": source_text,
                        "target_text": target_text,
                        "domain": domain,
                        "source_lang": self.source_lang,
                        "target_lang": self.target_lang,
                    }
                )

                self.rows_written += 1
                progress_bar.update(1)

                if self.rows_written >= self.max_samples:
                    break

            progress_bar.close()

        self.save_log()

        print("\nHugging Face corpus download completed successfully.")
        print(f"Rows seen    : {self.rows_seen}")
        print(f"Rows written : {self.rows_written}")
        print(f"Rows skipped : {self.rows_skipped}")
        print(f"Saved file   : {self.output_file}")

    def save_log(self):
        log_file = OUTPUT_DIR / "hf_parallel_download_log.json"

        log_data = {
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "max_samples": self.max_samples,
            "rows_seen": self.rows_seen,
            "rows_written": self.rows_written,
            "rows_skipped": self.rows_skipped,
            "output_file": str(self.output_file),
        }

        with open(log_file, "w", encoding="utf-8") as file:
            json.dump(log_data, file, indent=4, ensure_ascii=False)

        print(f"Download log saved at: {log_file}")


def run_hf_parallel_download():
    downloader = HuggingFaceParallelCorpusDownloader()
    downloader.download_and_save()


if __name__ == "__main__":
    run_hf_parallel_download()