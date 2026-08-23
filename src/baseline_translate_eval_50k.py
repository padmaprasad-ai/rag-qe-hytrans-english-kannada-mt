# src/baseline_translate_eval_50k.py

import json
import torch
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import (
    TEST_FILE_50K,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    BASELINE_TRANSLATION_FILE_50K,
    BASELINE_EVALUATION_REPORT_50K,
    BASELINE_SENTENCE_SCORES_50K,
    BASELINE_TRANSLATION_BATCH_SIZE_50K,
    BASELINE_TEST_LIMIT_50K,
    SAVE_EVERY_BATCH_50K,
    create_directories,
)


class BaselineTranslatorEvaluator50K:
    """
    Baseline pretrained NLLB translation and evaluation
    for the 50K English-Kannada experiment.

    Input:
        data/processed/test_50k.csv

    Outputs:
        outputs/baseline_50k_translations.csv
        outputs/baseline_50k_evaluation_report.json
        outputs/baseline_50k_sentence_scores.csv
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Step 28: Initializing 50K baseline translator.")
        print(f"Model       : {BASELINE_MODEL_NAME}")
        print(f"Device      : {self.device}")
        print(f"Source lang : {SOURCE_LANG_CODE}")
        print(f"Target lang : {TARGET_LANG_CODE}")

        if self.device == "cpu":
            print(
                "\nWARNING: CUDA GPU not detected. "
                "Translating 4,956 sentences on CPU may take considerable time.\n"
            )

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
            raise ValueError(f"Invalid target language code: {TARGET_LANG_CODE}")

        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

    def load_test_data(self):
        if not TEST_FILE_50K.exists():
            raise FileNotFoundError(
                f"50K test file not found:\n{TEST_FILE_50K}\n\n"
                "Please run Step 26 first: run_prepare_split_50k.py"
            )

        df = pd.read_csv(TEST_FILE_50K, encoding="utf-8-sig")

        required_columns = ["source_text", "target_text"]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=required_columns).reset_index(drop=True)

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()

        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]

        df = df.reset_index(drop=True)
        df.insert(0, "row_id", range(1, len(df) + 1))

        if BASELINE_TEST_LIMIT_50K is not None:
            df = df.head(int(BASELINE_TEST_LIMIT_50K)).copy()

        if len(df) == 0:
            raise ValueError("No valid rows found in test_50k.csv.")

        return df.reset_index(drop=True)

    def batch_translate(self, source_sentences):
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

        return [translation.strip() for translation in translations]

    def translate_test_set(self):
        create_directories()

        df = self.load_test_data()

        print("\n50K baseline translation started.")
        print(f"Total test sentences: {len(df)}")
        print(f"Batch size          : {BASELINE_TRANSLATION_BATCH_SIZE_50K}")
        print(f"Output file         : {BASELINE_TRANSLATION_FILE_50K}")

        output_rows = []

        source_sentences = df["source_text"].tolist()

        for start_idx in tqdm(
            range(0, len(source_sentences), BASELINE_TRANSLATION_BATCH_SIZE_50K),
            desc="Translating 50K test baseline",
        ):
            end_idx = start_idx + BASELINE_TRANSLATION_BATCH_SIZE_50K
            batch_sources = source_sentences[start_idx:end_idx]
            batch_df = df.iloc[start_idx:end_idx].copy()

            batch_predictions = self.batch_translate(batch_sources)

            for local_idx, (_, row) in enumerate(batch_df.iterrows()):
                output_rows.append(
                    {
                        "row_id": int(row["row_id"]),
                        "source_text": row["source_text"],
                        "target_text": row["target_text"],
                        "baseline_50k_prediction": batch_predictions[local_idx],
                        "baseline_model": BASELINE_MODEL_NAME,
                        "source_lang_code": SOURCE_LANG_CODE,
                        "target_lang_code": TARGET_LANG_CODE,
                    }
                )

            if SAVE_EVERY_BATCH_50K:
                temp_df = pd.DataFrame(output_rows)
                temp_df.to_csv(
                    BASELINE_TRANSLATION_FILE_50K,
                    index=False,
                    encoding="utf-8-sig",
                )

        output_df = pd.DataFrame(output_rows)

        output_df.to_csv(
            BASELINE_TRANSLATION_FILE_50K,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n50K baseline translation completed successfully.")
        print(f"Translation file saved at: {BASELINE_TRANSLATION_FILE_50K}")

        print("\nSample baseline translations:")
        for _, row in output_df.head(5).iterrows():
            print("\n----------------------------------------")
            print(f"Source    : {row['source_text']}")
            print(f"Reference : {row['target_text']}")
            print(f"Prediction: {row['baseline_50k_prediction']}")

        return output_df

    def evaluate(self, df):
        print("\nEvaluating 50K baseline translations...")

        references = df["target_text"].astype(str).str.strip().tolist()
        predictions = df["baseline_50k_prediction"].astype(str).str.strip().tolist()

        corpus_bleu = self.bleu.corpus_score(predictions, [references]).score
        corpus_chrf = self.chrf.corpus_score(predictions, [references]).score

        sentence_rows = []

        for index, row in df.iterrows():
            reference = str(row["target_text"]).strip()
            prediction = str(row["baseline_50k_prediction"]).strip()

            sent_bleu = self.bleu.sentence_score(prediction, [reference]).score
            sent_chrf = self.chrf.sentence_score(prediction, [reference]).score

            sentence_rows.append(
                {
                    "row_id": int(row["row_id"]),
                    "source_text": row["source_text"],
                    "reference_translation": reference,
                    "baseline_50k_prediction": prediction,
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
            "model": "50K Baseline NLLB",
            "base_model": BASELINE_MODEL_NAME,
            "source_lang_code": SOURCE_LANG_CODE,
            "target_lang_code": TARGET_LANG_CODE,
            "total_evaluated_sentences": int(len(df)),
            "metrics": {
                "BLEU": round(float(corpus_bleu), 4),
                "chrF++": round(float(corpus_chrf), 4),
            },
            "input_test_file": str(TEST_FILE_50K),
            "translation_file": str(BASELINE_TRANSLATION_FILE_50K),
        }

        with open(BASELINE_EVALUATION_REPORT_50K, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            BASELINE_SENTENCE_SCORES_50K,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n50K baseline evaluation completed successfully.")

        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nFiles saved:")
        print(f"Evaluation report: {BASELINE_EVALUATION_REPORT_50K}")
        print(f"Sentence scores  : {BASELINE_SENTENCE_SCORES_50K}")

        return report, sentence_scores_df

    def run(self):
        df = self.translate_test_set()
        return self.evaluate(df)


def run_baseline_translate_eval_50k():
    runner = BaselineTranslatorEvaluator50K()
    return runner.run()


if __name__ == "__main__":
    run_baseline_translate_eval_50k()