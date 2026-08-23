# src/large_baseline_translate_eval.py

import json
import torch
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import (
    TEST_LARGE_FILE,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    LARGE_TRANSLATION_BATCH_SIZE,
    LARGE_TEST_TRANSLATION_LIMIT,
    LARGE_BASELINE_TRANSLATION_FILE,
    LARGE_BASELINE_EVALUATION_REPORT,
    LARGE_BASELINE_SENTENCE_SCORES,
    create_directories,
)


class LargeBaselineTranslatorEvaluator:
    """
    Large-test baseline translation and evaluation module.

    It translates test_large.csv using pretrained NLLB and evaluates using:
        - BLEU
        - chrF++

    Input:
        data/processed/test_large.csv

    Outputs:
        outputs/large_baseline_translations.csv
        outputs/large_baseline_evaluation_report.json
        outputs/large_baseline_sentence_scores.csv
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Initializing large baseline translator...")
        print(f"Model       : {BASELINE_MODEL_NAME}")
        print(f"Device      : {self.device}")
        print(f"Source lang : {SOURCE_LANG_CODE}")
        print(f"Target lang : {TARGET_LANG_CODE}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASELINE_MODEL_NAME,
            src_lang=SOURCE_LANG_CODE,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(BASELINE_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            TARGET_LANG_CODE
        )

        if self.forced_bos_token_id is None:
            raise ValueError(
                f"Invalid target language code: {TARGET_LANG_CODE}"
            )

        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

    def load_test_data(self) -> pd.DataFrame:
        if not TEST_LARGE_FILE.exists():
            raise FileNotFoundError(
                f"Large test file not found:\n{TEST_LARGE_FILE}\n\n"
                "Please run Step 12 first: run_large_data_split.py"
            )

        df = pd.read_csv(TEST_LARGE_FILE, encoding="utf-8-sig")

        required_columns = ["source_text", "target_text"]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=["source_text", "target_text"]).reset_index(drop=True)

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()

        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]

        if LARGE_TEST_TRANSLATION_LIMIT is not None:
            df = df.head(int(LARGE_TEST_TRANSLATION_LIMIT)).copy()

        if len(df) == 0:
            raise ValueError("No valid rows found in test_large.csv.")

        return df.reset_index(drop=True)

    def batch_translate(self, source_sentences):
        """
        Translates a batch of source sentences.
        """
        self.tokenizer.src_lang = SOURCE_LANG_CODE

        inputs = self.tokenizer(
            source_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=MAX_TARGET_LENGTH,
                num_beams=5,
                early_stopping=True,
            )

        translations = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        translations = [translation.strip() for translation in translations]

        return translations

    def translate_test_set(self) -> pd.DataFrame:
        create_directories()

        df = self.load_test_data()

        print(f"\nLarge baseline translation started.")
        print(f"Total test sentences: {len(df)}")
        print(f"Batch size          : {LARGE_TRANSLATION_BATCH_SIZE}")

        source_sentences = df["source_text"].tolist()
        all_predictions = []

        for start_idx in tqdm(
            range(0, len(source_sentences), LARGE_TRANSLATION_BATCH_SIZE),
            desc="Translating large test set",
        ):
            batch = source_sentences[start_idx:start_idx + LARGE_TRANSLATION_BATCH_SIZE]
            batch_predictions = self.batch_translate(batch)
            all_predictions.extend(batch_predictions)

        df["large_baseline_prediction"] = all_predictions
        df["baseline_model"] = BASELINE_MODEL_NAME
        df["source_lang_code"] = SOURCE_LANG_CODE
        df["target_lang_code"] = TARGET_LANG_CODE

        LARGE_BASELINE_TRANSLATION_FILE.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            LARGE_BASELINE_TRANSLATION_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge baseline translation completed successfully.")
        print(f"Translation file saved at: {LARGE_BASELINE_TRANSLATION_FILE}")

        print("\nSample translations:")
        for _, row in df.head(5).iterrows():
            print("\n----------------------------------------")
            print(f"Source    : {row['source_text']}")
            print(f"Reference : {row['target_text']}")
            print(f"Prediction: {row['large_baseline_prediction']}")

        return df

    def evaluate(self, df: pd.DataFrame):
        print("\nEvaluating large baseline translations...")

        references = df["target_text"].astype(str).str.strip().tolist()
        predictions = df["large_baseline_prediction"].astype(str).str.strip().tolist()

        corpus_bleu = self.bleu.corpus_score(
            predictions,
            [references],
        ).score

        corpus_chrf = self.chrf.corpus_score(
            predictions,
            [references],
        ).score

        sentence_rows = []

        for index, row in df.iterrows():
            reference = str(row["target_text"]).strip()
            prediction = str(row["large_baseline_prediction"]).strip()

            sent_bleu = self.bleu.sentence_score(
                prediction,
                [reference],
            ).score

            sent_chrf = self.chrf.sentence_score(
                prediction,
                [reference],
            ).score

            sentence_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": reference,
                    "large_baseline_prediction": prediction,
                    "sentence_bleu": round(float(sent_bleu), 4),
                    "sentence_chrf++": round(float(sent_chrf), 4),
                    "reference_length_chars": len(reference),
                    "prediction_length_chars": len(prediction),
                    "reference_length_words": len(reference.split()),
                    "prediction_length_words": len(prediction.split()),
                }
            )

        sentence_scores_df = pd.DataFrame(sentence_rows)

        report = {
            "model": "Large Baseline NLLB",
            "base_model": BASELINE_MODEL_NAME,
            "source_lang_code": SOURCE_LANG_CODE,
            "target_lang_code": TARGET_LANG_CODE,
            "total_evaluated_sentences": len(df),
            "metrics": {
                "BLEU": round(float(corpus_bleu), 4),
                "chrF++": round(float(corpus_chrf), 4),
            },
            "input_test_file": str(TEST_LARGE_FILE),
            "translation_file": str(LARGE_BASELINE_TRANSLATION_FILE),
        }

        with open(LARGE_BASELINE_EVALUATION_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            LARGE_BASELINE_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge baseline evaluation completed successfully.")
        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nFiles saved:")
        print(f"Evaluation report: {LARGE_BASELINE_EVALUATION_REPORT}")
        print(f"Sentence scores  : {LARGE_BASELINE_SENTENCE_SCORES}")

        print("\nSample sentence scores:")
        print(sentence_scores_df.head(5).to_string(index=False))

        return report, sentence_scores_df

    def run(self):
        df = self.translate_test_set()
        return self.evaluate(df)


def run_large_baseline_translation_evaluation():
    runner = LargeBaselineTranslatorEvaluator()
    return runner.run()


if __name__ == "__main__":
    run_large_baseline_translation_evaluation()