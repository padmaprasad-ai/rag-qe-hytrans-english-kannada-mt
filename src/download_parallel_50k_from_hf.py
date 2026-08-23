# src/download_parallel_50k_from_hf.py

import csv
import json
from datasets import load_dataset
from tqdm import tqdm

from src.config import (
    HF_DATASET_NAME_50K,
    HF_DATASET_CONFIG_50K,
    HF_DATASET_SPLIT_50K,
    HF_MAX_SAMPLES_50K,
    HF_OUTPUT_FILE_50K,
    HF_SOURCE_ONLY_FILE_50K,
    HF_50K_DOWNLOAD_LOG,
    create_directories,
)


class HuggingFace50KDownloader:
    """
    Downloads 50K English-Kannada parallel sentence pairs
    from AI4Bharat Samanantar Hugging Face dataset.

    Output:
        data/raw/parallel_50k.csv
        data/raw/source_sentences_50k.txt
        outputs/hf_50k_download_log.json
    """

    def __init__(self):
        self.rows_seen = 0
        self.rows_written = 0
        self.rows_skipped = 0

    @staticmethod
    def clean_text(text):
        if text is None:
            return ""

        text = str(text).strip()
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")
        text = " ".join(text.split())

        return text

    def extract_pair(self, example):
        """
        Handles Samanantar format:
            src = English source sentence
            tgt = Kannada target sentence
        """

        if "src" in example and "tgt" in example:
            source_text = example["src"]
            target_text = example["tgt"]
            domain = example.get("data_source", "samanantar")
            return source_text, target_text, domain

        if "translation" in example:
            translation = example["translation"]

            source_text = (
                translation.get("en")
                or translation.get("eng")
                or translation.get("english")
            )

            target_text = (
                translation.get("kn")
                or translation.get("kan")
                or translation.get("kannada")
            )

            return source_text, target_text, "translation"

        return None, None, "unknown"

    def load_hf_dataset(self):
        print("Loading Hugging Face dataset in streaming mode...")
        print(f"Dataset : {HF_DATASET_NAME_50K}")
        print(f"Config  : {HF_DATASET_CONFIG_50K}")
        print(f"Split   : {HF_DATASET_SPLIT_50K}")

        try:
            dataset = load_dataset(
                HF_DATASET_NAME_50K,
                HF_DATASET_CONFIG_50K,
                split=HF_DATASET_SPLIT_50K,
                streaming=True,
                trust_remote_code=True,
            )
        except TypeError:
            dataset = load_dataset(
                HF_DATASET_NAME_50K,
                HF_DATASET_CONFIG_50K,
                split=HF_DATASET_SPLIT_50K,
                streaming=True,
            )

        return dataset

    def download(self):
        create_directories()

        HF_OUTPUT_FILE_50K.parent.mkdir(parents=True, exist_ok=True)
        HF_SOURCE_ONLY_FILE_50K.parent.mkdir(parents=True, exist_ok=True)
        HF_50K_DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)

        dataset = self.load_hf_dataset()

        print("\n50K corpus download started...")
        print(f"Target rows: {HF_MAX_SAMPLES_50K}")
        print(f"CSV output : {HF_OUTPUT_FILE_50K}")
        print(f"Source txt : {HF_SOURCE_ONLY_FILE_50K}")

        with open(HF_OUTPUT_FILE_50K, "w", encoding="utf-8-sig", newline="") as csv_file, \
             open(HF_SOURCE_ONLY_FILE_50K, "w", encoding="utf-8") as source_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "source_text",
                    "target_text",
                    "domain",
                    "source_lang",
                    "target_lang",
                ],
            )

            writer.writeheader()

            progress_bar = tqdm(total=HF_MAX_SAMPLES_50K, desc="Downloading 50K pairs")

            for example in dataset:
                self.rows_seen += 1

                source_text, target_text, domain = self.extract_pair(example)

                source_text = self.clean_text(source_text)
                target_text = self.clean_text(target_text)
                domain = self.clean_text(domain) or "samanantar"

                if not source_text or not target_text:
                    self.rows_skipped += 1
                    continue

                writer.writerow(
                    {
                        "source_text": source_text,
                        "target_text": target_text,
                        "domain": domain,
                        "source_lang": "en",
                        "target_lang": "kn",
                    }
                )

                source_file.write(source_text + "\n")

                self.rows_written += 1
                progress_bar.update(1)

                if self.rows_written >= HF_MAX_SAMPLES_50K:
                    break

            progress_bar.close()

        log_data = {
            "dataset_name": HF_DATASET_NAME_50K,
            "dataset_config": HF_DATASET_CONFIG_50K,
            "split": HF_DATASET_SPLIT_50K,
            "target_samples": HF_MAX_SAMPLES_50K,
            "rows_seen": self.rows_seen,
            "rows_written": self.rows_written,
            "rows_skipped": self.rows_skipped,
            "csv_output_file": str(HF_OUTPUT_FILE_50K),
            "source_only_file": str(HF_SOURCE_ONLY_FILE_50K),
        }

        with open(HF_50K_DOWNLOAD_LOG, "w", encoding="utf-8") as file:
            json.dump(log_data, file, indent=4, ensure_ascii=False)

        print("\nHugging Face 50K corpus download completed successfully.")
        print(f"Rows seen    : {self.rows_seen}")
        print(f"Rows written : {self.rows_written}")
        print(f"Rows skipped : {self.rows_skipped}")
        print(f"Saved CSV    : {HF_OUTPUT_FILE_50K}")
        print(f"Saved sources: {HF_SOURCE_ONLY_FILE_50K}")
        print(f"Log saved    : {HF_50K_DOWNLOAD_LOG}")


def run_hf_50k_download():
    downloader = HuggingFace50KDownloader()
    downloader.download()


if __name__ == "__main__":
    run_hf_50k_download()