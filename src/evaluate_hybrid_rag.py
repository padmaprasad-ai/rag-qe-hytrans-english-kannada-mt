# src/evaluate_hybrid_rag.py

import json
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from src.config import (
    HYBRID_TRANSLATION_FILE,
    HYBRID_EVALUATION_REPORT,
    HYBRID_SENTENCE_SCORES,
    create_directories,
)


class HybridRAGEvaluator:
    """
    Evaluates RAG-assisted hybrid translation using BLEU and chrF++.
    """

    def __init__(self):
        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

    def load_data(self):
        if not HYBRID_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Hybrid translation file not found:\n{HYBRID_TRANSLATION_FILE}\n"
                "Please run run_hybrid_rag_translation.py first."
            )

        df = pd.read_csv(HYBRID_TRANSLATION_FILE, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "hybrid_prediction",
            "selected_method",
            "retrieval_similarity",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=["target_text", "hybrid_prediction"])
        df["target_text"] = df["target_text"].astype(str).str.strip()
        df["hybrid_prediction"] = df["hybrid_prediction"].astype(str).str.strip()

        df = df[df["target_text"].str.len() > 0]
        df = df[df["hybrid_prediction"].str.len() > 0]

        if len(df) == 0:
            raise ValueError("No valid hybrid translations found for evaluation.")

        return df.reset_index(drop=True)

    def evaluate(self):
        create_directories()

        df = self.load_data()

        references = df["target_text"].tolist()
        predictions = df["hybrid_prediction"].tolist()

        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        chrf_score = self.chrf.corpus_score(predictions, [references]).score

        method_counts = df["selected_method"].value_counts().to_dict()

        report = {
            "total_evaluated_sentences": len(df),
            "metrics": {
                "BLEU": round(bleu_score, 4),
                "chrF++": round(chrf_score, 4),
            },
            "method_selection_counts": method_counts,
            "average_retrieval_similarity": round(
                float(df["retrieval_similarity"].mean()),
                4,
            ),
            "input_file": str(HYBRID_TRANSLATION_FILE),
        }

        sentence_rows = []

        for index, row in df.iterrows():
            sent_bleu = self.bleu.sentence_score(
                row["hybrid_prediction"],
                [row["target_text"]],
            ).score

            sent_chrf = self.chrf.sentence_score(
                row["hybrid_prediction"],
                [row["target_text"]],
            ).score

            sentence_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": row["target_text"],
                    "hybrid_prediction": row["hybrid_prediction"],
                    "selected_method": row["selected_method"],
                    "retrieval_similarity": row["retrieval_similarity"],
                    "sentence_bleu": round(sent_bleu, 4),
                    "sentence_chrf++": round(sent_chrf, 4),
                }
            )

        sentence_scores_df = pd.DataFrame(sentence_rows)

        with open(HYBRID_EVALUATION_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            HYBRID_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nHybrid RAG evaluation completed successfully.")
        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nMethod selection counts:")
        for method, count in method_counts.items():
            print(f"{method}: {count}")

        print(f"\nAverage retrieval similarity: {report['average_retrieval_similarity']}")

        print("\nFiles saved:")
        print(f"Hybrid evaluation report: {HYBRID_EVALUATION_REPORT}")
        print(f"Hybrid sentence scores  : {HYBRID_SENTENCE_SCORES}")

        print("\nSample scores:")
        print(sentence_scores_df.head(5).to_string(index=False))

        return report, sentence_scores_df


def run_hybrid_rag_evaluation():
    evaluator = HybridRAGEvaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    run_hybrid_rag_evaluation()