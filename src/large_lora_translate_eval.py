# src/large_lora_translate_eval.py

import json
import torch
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from src.config import (
    TEST_LARGE_FILE,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    LARGE_LORA_MODEL_DIR,
    LARGE_LORA_ADAPTER_DIR,
    LARGE_LORA_TRANSLATION_FILE,
    LARGE_LORA_EVALUATION_REPORT,
    LARGE_LORA_SENTENCE_SCORES,
    LARGE_BASELINE_TRANSLATION_FILE,
    LARGE_MODEL_COMPARISON_REPORT,
    LARGE_MODEL_COMPARISON_SENTENCE_SCORES,
    LARGE_LORA_TRANSLATION_BATCH_SIZE,
    LARGE_LORA_TEST_TRANSLATION_LIMIT,
    create_directories,
)


class LargeLoRATranslatorEvaluator:
    """
    Loads the base NLLB model + trained LoRA adapter,
    translates the large test set, evaluates BLEU/chrF++,
    and compares against the large baseline output.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Initializing large LoRA translator...")
        print(f"Base model     : {BASELINE_MODEL_NAME}")
        print(f"LoRA adapter   : {LARGE_LORA_ADAPTER_DIR}")
        print(f"Tokenizer path : {LARGE_LORA_MODEL_DIR}")
        print(f"Device         : {self.device}")
        print(f"Source lang    : {SOURCE_LANG_CODE}")
        print(f"Target lang    : {TARGET_LANG_CODE}")

        if not LARGE_LORA_ADAPTER_DIR.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found:\n{LARGE_LORA_ADAPTER_DIR}\n"
                "Please run Step 15 first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(LARGE_LORA_MODEL_DIR),
            src_lang=SOURCE_LANG_CODE,
        )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASELINE_MODEL_NAME)

        self.model = PeftModel.from_pretrained(
            base_model,
            str(LARGE_LORA_ADAPTER_DIR),
        )

        self.model.to(self.device)
        self.model.eval()

        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            TARGET_LANG_CODE
        )

        if self.forced_bos_token_id is None:
            raise ValueError(f"Invalid target language code: {TARGET_LANG_CODE}")

        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

    def load_test_data(self) -> pd.DataFrame:
        if not TEST_LARGE_FILE.exists():
            raise FileNotFoundError(
                f"Large test file not found:\n{TEST_LARGE_FILE}\n"
                "Please run Step 12 first."
            )

        df = pd.read_csv(TEST_LARGE_FILE, encoding="utf-8-sig")

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

        if LARGE_LORA_TEST_TRANSLATION_LIMIT is not None:
            df = df.head(int(LARGE_LORA_TEST_TRANSLATION_LIMIT)).copy()

        if len(df) == 0:
            raise ValueError("No valid rows found in test_large.csv.")

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

        print("\nLarge LoRA translation started.")
        print(f"Total test sentences: {len(df)}")
        print(f"Batch size          : {LARGE_LORA_TRANSLATION_BATCH_SIZE}")

        source_sentences = df["source_text"].tolist()
        predictions = []

        for start_idx in tqdm(
            range(0, len(source_sentences), LARGE_LORA_TRANSLATION_BATCH_SIZE),
            desc="Translating with LoRA model",
        ):
            batch = source_sentences[start_idx:start_idx + LARGE_LORA_TRANSLATION_BATCH_SIZE]
            batch_predictions = self.batch_translate(batch)
            predictions.extend(batch_predictions)

        df["large_lora_prediction"] = predictions
        df["base_model"] = BASELINE_MODEL_NAME
        df["lora_adapter"] = str(LARGE_LORA_ADAPTER_DIR)
        df["source_lang_code"] = SOURCE_LANG_CODE
        df["target_lang_code"] = TARGET_LANG_CODE

        df.to_csv(
            LARGE_LORA_TRANSLATION_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge LoRA translation completed successfully.")
        print(f"Translation file saved at: {LARGE_LORA_TRANSLATION_FILE}")

        print("\nSample LoRA translations:")
        for _, row in df.head(5).iterrows():
            print("\n----------------------------------------")
            print(f"Source    : {row['source_text']}")
            print(f"Reference : {row['target_text']}")
            print(f"Prediction: {row['large_lora_prediction']}")

        return df

    def evaluate_lora(self, df: pd.DataFrame):
        print("\nEvaluating large LoRA translations...")

        references = df["target_text"].astype(str).str.strip().tolist()
        predictions = df["large_lora_prediction"].astype(str).str.strip().tolist()

        corpus_bleu = self.bleu.corpus_score(predictions, [references]).score
        corpus_chrf = self.chrf.corpus_score(predictions, [references]).score

        sentence_rows = []

        for index, row in df.iterrows():
            reference = str(row["target_text"]).strip()
            prediction = str(row["large_lora_prediction"]).strip()

            sent_bleu = self.bleu.sentence_score(prediction, [reference]).score
            sent_chrf = self.chrf.sentence_score(prediction, [reference]).score

            sentence_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": reference,
                    "large_lora_prediction": prediction,
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
            "model": "Large LoRA NLLB",
            "base_model": BASELINE_MODEL_NAME,
            "adapter_path": str(LARGE_LORA_ADAPTER_DIR),
            "source_lang_code": SOURCE_LANG_CODE,
            "target_lang_code": TARGET_LANG_CODE,
            "total_evaluated_sentences": len(df),
            "metrics": {
                "BLEU": round(float(corpus_bleu), 4),
                "chrF++": round(float(corpus_chrf), 4),
            },
            "input_test_file": str(TEST_LARGE_FILE),
            "translation_file": str(LARGE_LORA_TRANSLATION_FILE),
        }

        with open(LARGE_LORA_EVALUATION_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            LARGE_LORA_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge LoRA evaluation completed successfully.")
        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nFiles saved:")
        print(f"Evaluation report: {LARGE_LORA_EVALUATION_REPORT}")
        print(f"Sentence scores  : {LARGE_LORA_SENTENCE_SCORES}")

        return report, sentence_scores_df

    def compare_with_baseline(self, lora_df: pd.DataFrame):
        if not LARGE_BASELINE_TRANSLATION_FILE.exists():
            print(
                "\nLarge baseline translation file not found. "
                "Skipping baseline vs LoRA comparison."
            )
            return None, None

        baseline_df = pd.read_csv(LARGE_BASELINE_TRANSLATION_FILE, encoding="utf-8-sig")

        required_baseline = ["source_text", "target_text", "large_baseline_prediction"]
        missing = [col for col in required_baseline if col not in baseline_df.columns]

        if missing:
            raise ValueError(
                f"Missing columns in large baseline file: {missing}\n"
                f"Detected columns: {list(baseline_df.columns)}"
            )

        compare_df = pd.merge(
            baseline_df[required_baseline],
            lora_df[["source_text", "large_lora_prediction"]],
            on="source_text",
            how="inner",
        )

        compare_df = compare_df.dropna(
            subset=[
                "target_text",
                "large_baseline_prediction",
                "large_lora_prediction",
            ]
        ).reset_index(drop=True)

        if len(compare_df) == 0:
            print("No matching rows found for baseline vs LoRA comparison.")
            return None, None

        references = compare_df["target_text"].astype(str).str.strip().tolist()
        baseline_predictions = compare_df["large_baseline_prediction"].astype(str).str.strip().tolist()
        lora_predictions = compare_df["large_lora_prediction"].astype(str).str.strip().tolist()

        baseline_bleu = self.bleu.corpus_score(baseline_predictions, [references]).score
        baseline_chrf = self.chrf.corpus_score(baseline_predictions, [references]).score

        lora_bleu = self.bleu.corpus_score(lora_predictions, [references]).score
        lora_chrf = self.chrf.corpus_score(lora_predictions, [references]).score

        comparison_rows = []

        for index, row in compare_df.iterrows():
            reference = str(row["target_text"]).strip()
            baseline_pred = str(row["large_baseline_prediction"]).strip()
            lora_pred = str(row["large_lora_prediction"]).strip()

            baseline_sent_bleu = self.bleu.sentence_score(baseline_pred, [reference]).score
            lora_sent_bleu = self.bleu.sentence_score(lora_pred, [reference]).score

            baseline_sent_chrf = self.chrf.sentence_score(baseline_pred, [reference]).score
            lora_sent_chrf = self.chrf.sentence_score(lora_pred, [reference]).score

            comparison_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": reference,
                    "large_baseline_prediction": baseline_pred,
                    "large_lora_prediction": lora_pred,
                    "baseline_sentence_bleu": round(float(baseline_sent_bleu), 4),
                    "lora_sentence_bleu": round(float(lora_sent_bleu), 4),
                    "bleu_difference": round(float(lora_sent_bleu - baseline_sent_bleu), 4),
                    "baseline_sentence_chrf++": round(float(baseline_sent_chrf), 4),
                    "lora_sentence_chrf++": round(float(lora_sent_chrf), 4),
                    "chrf++_difference": round(float(lora_sent_chrf - baseline_sent_chrf), 4),
                }
            )

        comparison_scores_df = pd.DataFrame(comparison_rows)

        report = {
            "total_compared_sentences": len(compare_df),
            "large_baseline_nllb": {
                "BLEU": round(float(baseline_bleu), 4),
                "chrF++": round(float(baseline_chrf), 4),
            },
            "large_lora_nllb": {
                "BLEU": round(float(lora_bleu), 4),
                "chrF++": round(float(lora_chrf), 4),
            },
            "absolute_improvement": {
                "BLEU": round(float(lora_bleu - baseline_bleu), 4),
                "chrF++": round(float(lora_chrf - baseline_chrf), 4),
            },
            "files": {
                "baseline_file": str(LARGE_BASELINE_TRANSLATION_FILE),
                "lora_file": str(LARGE_LORA_TRANSLATION_FILE),
            },
        }

        with open(LARGE_MODEL_COMPARISON_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        comparison_scores_df.to_csv(
            LARGE_MODEL_COMPARISON_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge baseline vs LoRA comparison completed successfully.")

        print("\nCorpus-level comparison:")
        print(f"Baseline BLEU      : {report['large_baseline_nllb']['BLEU']}")
        print(f"LoRA BLEU          : {report['large_lora_nllb']['BLEU']}")
        print(f"BLEU Improvement   : {report['absolute_improvement']['BLEU']}")

        print(f"\nBaseline chrF++    : {report['large_baseline_nllb']['chrF++']}")
        print(f"LoRA chrF++        : {report['large_lora_nllb']['chrF++']}")
        print(f"chrF++ Improvement : {report['absolute_improvement']['chrF++']}")

        print("\nFiles saved:")
        print(f"Comparison report: {LARGE_MODEL_COMPARISON_REPORT}")
        print(f"Sentence scores  : {LARGE_MODEL_COMPARISON_SENTENCE_SCORES}")

        print("\nSample comparison:")
        print(comparison_scores_df.head(5).to_string(index=False))

        return report, comparison_scores_df

    def run(self):
        lora_df = self.translate_test_set()
        lora_report, lora_sentence_scores = self.evaluate_lora(lora_df)
        comparison_report, comparison_scores = self.compare_with_baseline(lora_df)

        return lora_report, lora_sentence_scores, comparison_report, comparison_scores


def run_large_lora_translation_evaluation():
    runner = LargeLoRATranslatorEvaluator()
    return runner.run()


if __name__ == "__main__":
    run_large_lora_translation_evaluation()