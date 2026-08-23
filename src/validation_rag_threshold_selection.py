# src/validation_rag_threshold_selection.py

import json
import torch
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from src.config import (
    VALID_LARGE_FILE,
    LARGE_RAG_TRANSLATION_FILE,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    LARGE_LORA_MODEL_DIR,
    LARGE_LORA_ADAPTER_DIR,
    VALIDATION_RAG_CANDIDATES_FILE,
    VALIDATION_RAG_THRESHOLD_TUNING_CSV,
    VALIDATION_RAG_THRESHOLD_SELECTION_REPORT,
    TEST_VALIDATION_SELECTED_RAG_TRANSLATION_FILE,
    TEST_VALIDATION_SELECTED_RAG_EVALUATION_REPORT,
    TEST_VALIDATION_SELECTED_RAG_SENTENCE_SCORES,
    VALIDATION_RAG_TRANSLATION_BATCH_SIZE,
    VALIDATION_RAG_TRANSLATION_LIMIT,
    VALIDATION_RAG_TOP_K,
    VALIDATION_RAG_THRESHOLDS,
    REUSE_VALIDATION_RAG_CANDIDATES,
    create_directories,
)

from src.build_large_retriever import LargeTranslationMemoryRetriever


class ValidationRAGThresholdSelector:
    """
    Selects RAG threshold using validation set, then applies the selected
    threshold to the test set.

    This avoids selecting the RAG threshold directly on test data.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

        print("Step 22: Validation-based RAG threshold selection started.")
        print(f"Device selected       : {self.device}")
        print(f"Base model            : {BASELINE_MODEL_NAME}")
        print(f"LoRA adapter          : {LARGE_LORA_ADAPTER_DIR}")
        print(f"Validation file       : {VALID_LARGE_FILE}")
        print(f"Thresholds            : {VALIDATION_RAG_THRESHOLDS}")

        self.tokenizer = None
        self.model = None
        self.retriever = None
        self.forced_bos_token_id = None

    def load_lora_model(self):
        print("\nLoading LoRA model for validation translation...")

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

        print("LoRA model loaded successfully.")

    def load_retriever(self):
        print("\nLoading large retriever...")

        self.retriever = LargeTranslationMemoryRetriever()
        self.retriever.load()

        print("Large retriever loaded successfully.")

    def load_validation_data(self):
        if not VALID_LARGE_FILE.exists():
            raise FileNotFoundError(
                f"Validation file not found:\n{VALID_LARGE_FILE}\n"
                "Please run Step 12 first."
            )

        df = pd.read_csv(VALID_LARGE_FILE, encoding="utf-8-sig")

        required_columns = ["source_text", "target_text"]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing columns in validation file: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=required_columns).reset_index(drop=True)

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()

        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]

        if VALIDATION_RAG_TRANSLATION_LIMIT is not None:
            df = df.head(int(VALIDATION_RAG_TRANSLATION_LIMIT)).copy()

        if len(df) == 0:
            raise ValueError("No valid validation rows found.")

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

    def create_validation_candidates(self):
        create_directories()

        if (
            REUSE_VALIDATION_RAG_CANDIDATES
            and VALIDATION_RAG_CANDIDATES_FILE.exists()
        ):
            print("\nReusing existing validation RAG candidates.")
            print(f"File: {VALIDATION_RAG_CANDIDATES_FILE}")
            return pd.read_csv(VALIDATION_RAG_CANDIDATES_FILE, encoding="utf-8-sig")

        self.load_lora_model()
        self.load_retriever()

        df = self.load_validation_data()

        print("\nGenerating validation LoRA predictions...")
        print(f"Validation sentences: {len(df)}")
        print(f"Batch size          : {VALIDATION_RAG_TRANSLATION_BATCH_SIZE}")

        source_sentences = df["source_text"].tolist()
        lora_predictions = []

        for start_idx in tqdm(
            range(0, len(source_sentences), VALIDATION_RAG_TRANSLATION_BATCH_SIZE),
            desc="Validation LoRA translation",
        ):
            batch = source_sentences[start_idx:start_idx + VALIDATION_RAG_TRANSLATION_BATCH_SIZE]
            batch_predictions = self.batch_translate(batch)
            lora_predictions.extend(batch_predictions)

        df["validation_lora_prediction"] = lora_predictions

        print("\nRetrieving validation translation-memory candidates...")

        rows = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Validation retrieval",
        ):
            source_text = str(row["source_text"]).strip()
            reference_text = str(row["target_text"]).strip()
            lora_prediction = str(row["validation_lora_prediction"]).strip()

            retrieved = self.retriever.retrieve(
                query=source_text,
                top_k=VALIDATION_RAG_TOP_K,
            )

            if retrieved:
                best = retrieved[0]
                retrieved_source = best["source_text"]
                retrieved_translation = best["target_text"]
                retrieval_similarity = float(best["similarity_score"])
            else:
                retrieved_source = ""
                retrieved_translation = ""
                retrieval_similarity = 0.0

            rows.append(
                {
                    "source_text": source_text,
                    "target_text": reference_text,
                    "validation_lora_prediction": lora_prediction,
                    "retrieved_source": retrieved_source,
                    "retrieved_translation": retrieved_translation,
                    "retrieval_similarity": round(float(retrieval_similarity), 4),
                }
            )

        candidate_df = pd.DataFrame(rows)

        candidate_df.to_csv(
            VALIDATION_RAG_CANDIDATES_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nValidation RAG candidates saved successfully.")
        print(f"File: {VALIDATION_RAG_CANDIDATES_FILE}")

        print("\nSample validation candidates:")
        print(
            candidate_df[
                [
                    "source_text",
                    "validation_lora_prediction",
                    "retrieval_similarity",
                    "retrieved_translation",
                ]
            ].head(5).to_string(index=False)
        )

        return candidate_df

    @staticmethod
    def choose_prediction(row, threshold, lora_column):
        if float(row["retrieval_similarity"]) >= threshold:
            return str(row["retrieved_translation"]).strip(), "retrieval_memory"
        return str(row[lora_column]).strip(), "large_lora_nllb"

    def evaluate_threshold_on_validation(self, df, threshold):
        predictions = []
        selected_methods = []

        for _, row in df.iterrows():
            prediction, method = self.choose_prediction(
                row=row,
                threshold=threshold,
                lora_column="validation_lora_prediction",
            )
            predictions.append(prediction)
            selected_methods.append(method)

        references = df["target_text"].astype(str).str.strip().tolist()

        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        chrf_score = self.chrf.corpus_score(predictions, [references]).score

        return {
            "threshold": float(threshold),
            "validation_BLEU": round(float(bleu_score), 4),
            "validation_chrF++": round(float(chrf_score), 4),
            "retrieval_selected": int(selected_methods.count("retrieval_memory")),
            "lora_selected": int(selected_methods.count("large_lora_nllb")),
            "total_sentences": int(len(df)),
        }

    def tune_threshold_on_validation(self, validation_df):
        print("\nTuning RAG threshold on validation set...")

        results = []

        for threshold in VALIDATION_RAG_THRESHOLDS:
            result = self.evaluate_threshold_on_validation(validation_df, threshold)
            results.append(result)

        results_df = pd.DataFrame(results)

        results_df.to_csv(
            VALIDATION_RAG_THRESHOLD_TUNING_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        best_bleu_row = results_df.sort_values(
            by=["validation_BLEU", "validation_chrF++"],
            ascending=False,
        ).iloc[0]

        best_chrf_row = results_df.sort_values(
            by=["validation_chrF++", "validation_BLEU"],
            ascending=False,
        ).iloc[0]

        selected_threshold = float(best_bleu_row["threshold"])

        selection_report = {
            "selection_method": "validation_best_bleu",
            "selected_threshold": selected_threshold,
            "best_by_validation_bleu": {
                "threshold": float(best_bleu_row["threshold"]),
                "validation_BLEU": float(best_bleu_row["validation_BLEU"]),
                "validation_chrF++": float(best_bleu_row["validation_chrF++"]),
                "retrieval_selected": int(best_bleu_row["retrieval_selected"]),
                "lora_selected": int(best_bleu_row["lora_selected"]),
            },
            "best_by_validation_chrf++": {
                "threshold": float(best_chrf_row["threshold"]),
                "validation_BLEU": float(best_chrf_row["validation_BLEU"]),
                "validation_chrF++": float(best_chrf_row["validation_chrF++"]),
                "retrieval_selected": int(best_chrf_row["retrieval_selected"]),
                "lora_selected": int(best_chrf_row["lora_selected"]),
            },
            "threshold_tuning_csv": str(VALIDATION_RAG_THRESHOLD_TUNING_CSV),
        }

        with open(
            VALIDATION_RAG_THRESHOLD_SELECTION_REPORT,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(selection_report, file, indent=4, ensure_ascii=False)

        print("\nValidation threshold tuning completed successfully.")
        print("\nValidation threshold tuning results:")
        print(results_df.to_string(index=False))

        print("\nSelected threshold using validation BLEU:")
        print(selected_threshold)

        print("\nBest by validation BLEU:")
        print(selection_report["best_by_validation_bleu"])

        print("\nBest by validation chrF++:")
        print(selection_report["best_by_validation_chrf++"])

        print("\nFiles saved:")
        print(f"Validation tuning CSV : {VALIDATION_RAG_THRESHOLD_TUNING_CSV}")
        print(f"Selection report      : {VALIDATION_RAG_THRESHOLD_SELECTION_REPORT}")

        return selected_threshold, results_df, selection_report

    def load_test_candidates(self):
        if not LARGE_RAG_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Test RAG candidate file not found:\n{LARGE_RAG_TRANSLATION_FILE}\n"
                "Please run Step 17 first."
            )

        df = pd.read_csv(LARGE_RAG_TRANSLATION_FILE, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "large_lora_prediction",
            "retrieved_translation",
            "retrieval_similarity",
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing columns in test RAG file: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=required_columns).reset_index(drop=True)

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()
        df["large_lora_prediction"] = df["large_lora_prediction"].astype(str).str.strip()
        df["retrieved_translation"] = df["retrieved_translation"].astype(str).str.strip()

        df["retrieval_similarity"] = pd.to_numeric(
            df["retrieval_similarity"],
            errors="coerce",
        )

        df = df.dropna(subset=["retrieval_similarity"])

        return df.reset_index(drop=True)

    def apply_selected_threshold_to_test(self, selected_threshold):
        print("\nApplying validation-selected threshold to test set...")
        print(f"Selected threshold: {selected_threshold}")

        test_df = self.load_test_candidates()

        rows = []

        for _, row in test_df.iterrows():
            prediction, method = self.choose_prediction(
                row=row,
                threshold=selected_threshold,
                lora_column="large_lora_prediction",
            )

            rows.append(
                {
                    "source_text": row["source_text"],
                    "target_text": row["target_text"],
                    "large_lora_prediction": row["large_lora_prediction"],
                    "retrieved_translation": row["retrieved_translation"],
                    "retrieval_similarity": round(float(row["retrieval_similarity"]), 4),
                    "validation_selected_rag_prediction": prediction,
                    "selected_method": method,
                    "selected_threshold": selected_threshold,
                }
            )

        output_df = pd.DataFrame(rows)

        output_df.to_csv(
            TEST_VALIDATION_SELECTED_RAG_TRANSLATION_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        references = output_df["target_text"].astype(str).str.strip().tolist()
        predictions = output_df["validation_selected_rag_prediction"].astype(str).str.strip().tolist()

        test_bleu = self.bleu.corpus_score(predictions, [references]).score
        test_chrf = self.chrf.corpus_score(predictions, [references]).score

        method_counts = output_df["selected_method"].value_counts().to_dict()

        sentence_rows = []

        for index, row in output_df.iterrows():
            reference = str(row["target_text"]).strip()
            prediction = str(row["validation_selected_rag_prediction"]).strip()

            sent_bleu = self.bleu.sentence_score(prediction, [reference]).score
            sent_chrf = self.chrf.sentence_score(prediction, [reference]).score

            sentence_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": reference,
                    "large_lora_prediction": row["large_lora_prediction"],
                    "retrieved_translation": row["retrieved_translation"],
                    "validation_selected_rag_prediction": prediction,
                    "selected_method": row["selected_method"],
                    "retrieval_similarity": row["retrieval_similarity"],
                    "sentence_bleu": round(float(sent_bleu), 4),
                    "sentence_chrf++": round(float(sent_chrf), 4),
                }
            )

        sentence_scores_df = pd.DataFrame(sentence_rows)

        sentence_scores_df.to_csv(
            TEST_VALIDATION_SELECTED_RAG_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        final_report = {
            "model": "Validation-Selected RAG-Hybrid",
            "threshold_selection": "selected on validation set using BLEU",
            "selected_threshold": float(selected_threshold),
            "total_test_sentences": int(len(output_df)),
            "test_metrics": {
                "BLEU": round(float(test_bleu), 4),
                "chrF++": round(float(test_chrf), 4),
            },
            "method_selection_counts": {
                key: int(value) for key, value in method_counts.items()
            },
            "average_retrieval_similarity": round(
                float(output_df["retrieval_similarity"].mean()),
                4,
            ),
            "files": {
                "test_translation_file": str(TEST_VALIDATION_SELECTED_RAG_TRANSLATION_FILE),
                "test_sentence_scores": str(TEST_VALIDATION_SELECTED_RAG_SENTENCE_SCORES),
                "validation_threshold_report": str(VALIDATION_RAG_THRESHOLD_SELECTION_REPORT),
            },
        }

        with open(
            TEST_VALIDATION_SELECTED_RAG_EVALUATION_REPORT,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(final_report, file, indent=4, ensure_ascii=False)

        print("\nValidation-selected test evaluation completed successfully.")

        print("\nFinal test results:")
        print(f"Selected threshold : {selected_threshold}")
        print(f"BLEU               : {final_report['test_metrics']['BLEU']}")
        print(f"chrF++             : {final_report['test_metrics']['chrF++']}")

        print("\nMethod selection counts:")
        for method, count in method_counts.items():
            print(f"{method}: {count}")

        print("\nFiles saved:")
        print(f"Test translation file : {TEST_VALIDATION_SELECTED_RAG_TRANSLATION_FILE}")
        print(f"Test evaluation report: {TEST_VALIDATION_SELECTED_RAG_EVALUATION_REPORT}")
        print(f"Test sentence scores  : {TEST_VALIDATION_SELECTED_RAG_SENTENCE_SCORES}")

        return final_report, output_df, sentence_scores_df

    def run(self):
        validation_candidates_df = self.create_validation_candidates()
        selected_threshold, validation_results_df, selection_report = (
            self.tune_threshold_on_validation(validation_candidates_df)
        )

        final_report, test_output_df, sentence_scores_df = (
            self.apply_selected_threshold_to_test(selected_threshold)
        )

        return {
            "validation_threshold_results": validation_results_df,
            "selection_report": selection_report,
            "final_test_report": final_report,
        }


def run_validation_rag_threshold_selection():
    selector = ValidationRAGThresholdSelector()
    return selector.run()


if __name__ == "__main__":
    run_validation_rag_threshold_selection()