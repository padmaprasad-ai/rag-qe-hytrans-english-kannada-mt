# src/tune_large_rag_threshold.py

import json
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from src.config import (
    LARGE_RAG_TRANSLATION_FILE,
    LARGE_RAG_THRESHOLD_TUNING_CSV,
    LARGE_RAG_THRESHOLD_TUNING_REPORT,
    create_directories,
)


class LargeRAGThresholdTuner:
    """
    Tunes retrieval similarity threshold for large RAG-Hybrid MT.

    It reuses:
        - large_lora_prediction
        - retrieved_translation
        - retrieval_similarity
        - target_text

    It does not reload NLLB or LoRA model.
    """

    def __init__(self):
        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

        self.thresholds = [
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ]

    def load_large_rag_file(self):
        if not LARGE_RAG_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Large RAG translation file not found:\n{LARGE_RAG_TRANSLATION_FILE}\n\n"
                "Please run Step 17 first: run_large_rag_hybrid_eval.py"
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
                f"Missing required columns: {missing}\n"
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

        df = df[df["target_text"].str.len() > 0]
        df = df[df["large_lora_prediction"].str.len() > 0]
        df = df[df["retrieved_translation"].str.len() > 0]

        if len(df) == 0:
            raise ValueError("No valid rows found for large RAG threshold tuning.")

        return df.reset_index(drop=True)

    def select_prediction(self, row, threshold):
        if row["retrieval_similarity"] >= threshold:
            return row["retrieved_translation"], "retrieval_memory"
        else:
            return row["large_lora_prediction"], "large_lora_nllb"

    def evaluate_threshold(self, df, threshold):
        predictions = []
        selected_methods = []

        for _, row in df.iterrows():
            prediction, method = self.select_prediction(row, threshold)
            predictions.append(prediction)
            selected_methods.append(method)

        references = df["target_text"].astype(str).tolist()

        bleu_score = self.bleu.corpus_score(
            predictions,
            [references],
        ).score

        chrf_score = self.chrf.corpus_score(
            predictions,
            [references],
        ).score

        retrieval_selected = selected_methods.count("retrieval_memory")
        lora_selected = selected_methods.count("large_lora_nllb")

        return {
            "threshold": threshold,
            "BLEU": round(float(bleu_score), 4),
            "chrF++": round(float(chrf_score), 4),
            "retrieval_selected": retrieval_selected,
            "lora_selected": lora_selected,
            "total_sentences": len(df),
        }

    def tune(self):
        create_directories()

        print("Loading large RAG-Hybrid translation file...")
        df = self.load_large_rag_file()

        print(f"Rows available for tuning: {len(df)}")
        print("Running large RAG threshold tuning...")

        results = []

        for threshold in self.thresholds:
            result = self.evaluate_threshold(df, threshold)
            results.append(result)

        results_df = pd.DataFrame(results)

        best_bleu_row = results_df.sort_values(
            by=["BLEU", "chrF++"],
            ascending=False,
        ).iloc[0]

        best_chrf_row = results_df.sort_values(
            by=["chrF++", "BLEU"],
            ascending=False,
        ).iloc[0]

        no_retrieval_row = results_df[results_df["retrieval_selected"] == 0]

        if len(no_retrieval_row) > 0:
            pure_lora_reference = no_retrieval_row.iloc[0].to_dict()
        else:
            pure_lora_reference = None

        report = {
            "total_sentences": len(df),
            "thresholds_tested": self.thresholds,
            "best_by_bleu": {
                "threshold": float(best_bleu_row["threshold"]),
                "BLEU": float(best_bleu_row["BLEU"]),
                "chrF++": float(best_bleu_row["chrF++"]),
                "retrieval_selected": int(best_bleu_row["retrieval_selected"]),
                "lora_selected": int(best_bleu_row["lora_selected"]),
            },
            "best_by_chrf++": {
                "threshold": float(best_chrf_row["threshold"]),
                "BLEU": float(best_chrf_row["BLEU"]),
                "chrF++": float(best_chrf_row["chrF++"]),
                "retrieval_selected": int(best_chrf_row["retrieval_selected"]),
                "lora_selected": int(best_chrf_row["lora_selected"]),
            },
            "pure_lora_reference": pure_lora_reference,
        }

        results_df.to_csv(
            LARGE_RAG_THRESHOLD_TUNING_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        with open(LARGE_RAG_THRESHOLD_TUNING_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        print("\nLarge RAG threshold tuning completed successfully.")

        print("\nThreshold tuning results:")
        print(results_df.to_string(index=False))

        print("\nBest threshold by BLEU:")
        print(report["best_by_bleu"])

        print("\nBest threshold by chrF++:")
        print(report["best_by_chrf++"])

        print("\nFiles saved:")
        print(f"Tuning CSV   : {LARGE_RAG_THRESHOLD_TUNING_CSV}")
        print(f"Tuning report: {LARGE_RAG_THRESHOLD_TUNING_REPORT}")

        return report, results_df


def run_large_rag_threshold_tuning():
    tuner = LargeRAGThresholdTuner()
    return tuner.tune()


if __name__ == "__main__":
    run_large_rag_threshold_tuning()