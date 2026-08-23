# src/compare_models.py

import json
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from src.config import (
    BASELINE_TRANSLATION_FILE,
    FINETUNED_TRANSLATION_FILE,
    MODEL_COMPARISON_REPORT,
    MODEL_COMPARISON_SENTENCE_SCORES,
    create_directories,
)


class ModelComparisonEvaluator:
    """
    Compares baseline NLLB and fine-tuned NLLB using BLEU and chrF++.
    """

    def __init__(self):
        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

    def load_files(self):
        if not BASELINE_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Baseline translation file not found:\n{BASELINE_TRANSLATION_FILE}"
            )

        if not FINETUNED_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Fine-tuned translation file not found:\n{FINETUNED_TRANSLATION_FILE}"
            )

        baseline_df = pd.read_csv(BASELINE_TRANSLATION_FILE, encoding="utf-8-sig")
        finetuned_df = pd.read_csv(FINETUNED_TRANSLATION_FILE, encoding="utf-8-sig")

        required_baseline = ["source_text", "target_text", "baseline_prediction"]
        required_finetuned = ["source_text", "target_text", "finetuned_prediction"]

        for col in required_baseline:
            if col not in baseline_df.columns:
                raise ValueError(f"Missing column in baseline file: {col}")

        for col in required_finetuned:
            if col not in finetuned_df.columns:
                raise ValueError(f"Missing column in fine-tuned file: {col}")

        baseline_df = baseline_df[required_baseline].copy()
        finetuned_df = finetuned_df[required_finetuned].copy()

        merged_df = pd.merge(
            baseline_df,
            finetuned_df[["source_text", "finetuned_prediction"]],
            on="source_text",
            how="inner",
        )

        merged_df = merged_df.dropna(
            subset=[
                "source_text",
                "target_text",
                "baseline_prediction",
                "finetuned_prediction",
            ]
        )

        if len(merged_df) == 0:
            raise ValueError("No matching source sentences found between baseline and fine-tuned outputs.")

        return merged_df.reset_index(drop=True)

    def corpus_score(self, predictions, references):
        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        chrf_score = self.chrf.corpus_score(predictions, [references]).score

        return {
            "BLEU": round(bleu_score, 4),
            "chrF++": round(chrf_score, 4),
        }

    def sentence_score(self, prediction, reference):
        bleu_score = self.bleu.sentence_score(prediction, [reference]).score
        chrf_score = self.chrf.sentence_score(prediction, [reference]).score

        return round(bleu_score, 4), round(chrf_score, 4)

    def compare(self):
        create_directories()

        print("Loading baseline and fine-tuned translation files...")
        df = self.load_files()

        references = df["target_text"].astype(str).tolist()
        baseline_predictions = df["baseline_prediction"].astype(str).tolist()
        finetuned_predictions = df["finetuned_prediction"].astype(str).tolist()

        print(f"Matched sentence pairs: {len(df)}")

        baseline_scores = self.corpus_score(baseline_predictions, references)
        finetuned_scores = self.corpus_score(finetuned_predictions, references)

        report = {
            "total_compared_sentences": len(df),
            "baseline_nllb": baseline_scores,
            "finetuned_nllb": finetuned_scores,
            "absolute_improvement": {
                "BLEU": round(finetuned_scores["BLEU"] - baseline_scores["BLEU"], 4),
                "chrF++": round(finetuned_scores["chrF++"] - baseline_scores["chrF++"], 4),
            },
            "input_files": {
                "baseline_file": str(BASELINE_TRANSLATION_FILE),
                "finetuned_file": str(FINETUNED_TRANSLATION_FILE),
            },
        }

        sentence_rows = []

        for idx, row in df.iterrows():
            baseline_bleu, baseline_chrf = self.sentence_score(
                row["baseline_prediction"],
                row["target_text"],
            )

            finetuned_bleu, finetuned_chrf = self.sentence_score(
                row["finetuned_prediction"],
                row["target_text"],
            )

            sentence_rows.append(
                {
                    "id": idx + 1,
                    "source_text": row["source_text"],
                    "reference_translation": row["target_text"],
                    "baseline_prediction": row["baseline_prediction"],
                    "finetuned_prediction": row["finetuned_prediction"],
                    "baseline_sentence_bleu": baseline_bleu,
                    "finetuned_sentence_bleu": finetuned_bleu,
                    "bleu_difference": round(finetuned_bleu - baseline_bleu, 4),
                    "baseline_sentence_chrf++": baseline_chrf,
                    "finetuned_sentence_chrf++": finetuned_chrf,
                    "chrf++_difference": round(finetuned_chrf - baseline_chrf, 4),
                }
            )

        sentence_scores_df = pd.DataFrame(sentence_rows)

        MODEL_COMPARISON_REPORT.parent.mkdir(parents=True, exist_ok=True)

        with open(MODEL_COMPARISON_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            MODEL_COMPARISON_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nComparison completed successfully.")

        print("\nCorpus-level comparison:")
        print(f"Baseline BLEU       : {baseline_scores['BLEU']}")
        print(f"Fine-tuned BLEU     : {finetuned_scores['BLEU']}")
        print(f"BLEU Improvement    : {report['absolute_improvement']['BLEU']}")

        print(f"\nBaseline chrF++     : {baseline_scores['chrF++']}")
        print(f"Fine-tuned chrF++   : {finetuned_scores['chrF++']}")
        print(f"chrF++ Improvement  : {report['absolute_improvement']['chrF++']}")

        print("\nFiles saved:")
        print(f"Comparison report: {MODEL_COMPARISON_REPORT}")
        print(f"Sentence scores  : {MODEL_COMPARISON_SENTENCE_SCORES}")

        print("\nSample comparison:")
        print(sentence_scores_df.head(5).to_string(index=False))

        return report, sentence_scores_df


def run_model_comparison():
    evaluator = ModelComparisonEvaluator()
    return evaluator.compare()


if __name__ == "__main__":
    run_model_comparison()