# src/tune_rag_threshold.py

import json
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from src.config import (
    HYBRID_TRANSLATION_FILE,
    RAG_THRESHOLD_TUNING_REPORT,
    RAG_THRESHOLD_TUNING_CSV,
    create_directories,
)


class RAGThresholdTuner:
    """
    Tunes the RAG similarity threshold using already generated hybrid outputs.

    It does not rerun the NMT model.
    It reuses:
        nmt_prediction
        retrieved_translation
        retrieval_similarity

    Then it simulates different thresholds and evaluates BLEU and chrF++.
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
            0.88,
            0.90,
            0.95,
        ]

    def load_hybrid_file(self):
        if not HYBRID_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Hybrid translation file not found:\n{HYBRID_TRANSLATION_FILE}\n"
                "Please run run_hybrid_rag_translation.py first."
            )

        df = pd.read_csv(HYBRID_TRANSLATION_FILE, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "nmt_prediction",
            "retrieved_translation",
            "retrieval_similarity",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(
            subset=[
                "source_text",
                "target_text",
                "nmt_prediction",
                "retrieved_translation",
                "retrieval_similarity",
            ]
        )

        df["target_text"] = df["target_text"].astype(str).str.strip()
        df["nmt_prediction"] = df["nmt_prediction"].astype(str).str.strip()
        df["retrieved_translation"] = df["retrieved_translation"].astype(str).str.strip()
        df["retrieval_similarity"] = pd.to_numeric(
            df["retrieval_similarity"],
            errors="coerce",
        )

        df = df.dropna(subset=["retrieval_similarity"])

        if len(df) == 0:
            raise ValueError("No valid rows found for RAG threshold tuning.")

        return df.reset_index(drop=True)

    def select_prediction_by_threshold(self, row, threshold):
        if row["retrieval_similarity"] >= threshold:
            return row["retrieved_translation"], "retrieval_memory"
        else:
            return row["nmt_prediction"], "finetuned_nllb"

    def evaluate_threshold(self, df, threshold):
        predictions = []
        methods = []

        for _, row in df.iterrows():
            prediction, method = self.select_prediction_by_threshold(row, threshold)
            predictions.append(prediction)
            methods.append(method)

        references = df["target_text"].tolist()

        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        chrf_score = self.chrf.corpus_score(predictions, [references]).score

        retrieval_count = methods.count("retrieval_memory")
        nmt_count = methods.count("finetuned_nllb")

        return {
            "threshold": threshold,
            "BLEU": round(bleu_score, 4),
            "chrF++": round(chrf_score, 4),
            "retrieval_selected": retrieval_count,
            "nmt_selected": nmt_count,
            "total_sentences": len(df),
        }

    def tune(self):
        create_directories()

        print("Loading hybrid RAG output file...")
        df = self.load_hybrid_file()

        print(f"Rows available for threshold tuning: {len(df)}")
        print("Running threshold tuning...")

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

        report = {
            "total_sentences": len(df),
            "thresholds_tested": self.thresholds,
            "best_by_bleu": {
                "threshold": float(best_bleu_row["threshold"]),
                "BLEU": float(best_bleu_row["BLEU"]),
                "chrF++": float(best_bleu_row["chrF++"]),
                "retrieval_selected": int(best_bleu_row["retrieval_selected"]),
                "nmt_selected": int(best_bleu_row["nmt_selected"]),
            },
            "best_by_chrf++": {
                "threshold": float(best_chrf_row["threshold"]),
                "BLEU": float(best_chrf_row["BLEU"]),
                "chrF++": float(best_chrf_row["chrF++"]),
                "retrieval_selected": int(best_chrf_row["retrieval_selected"]),
                "nmt_selected": int(best_chrf_row["nmt_selected"]),
            },
        }

        results_df.to_csv(
            RAG_THRESHOLD_TUNING_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        with open(RAG_THRESHOLD_TUNING_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        print("\nRAG threshold tuning completed successfully.")
        print("\nThreshold tuning results:")
        print(results_df.to_string(index=False))

        print("\nBest threshold by BLEU:")
        print(report["best_by_bleu"])

        print("\nBest threshold by chrF++:")
        print(report["best_by_chrf++"])

        print("\nFiles saved:")
        print(f"Tuning CSV   : {RAG_THRESHOLD_TUNING_CSV}")
        print(f"Tuning report: {RAG_THRESHOLD_TUNING_REPORT}")

        return report, results_df


def run_rag_threshold_tuning():
    tuner = RAGThresholdTuner()
    return tuner.tune()


if __name__ == "__main__":
    run_rag_threshold_tuning()