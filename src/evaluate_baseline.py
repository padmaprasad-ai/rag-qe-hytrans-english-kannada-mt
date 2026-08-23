# src/evaluate_baseline.py

import json
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from src.config import (
    BASELINE_TRANSLATION_FILE,
    BASELINE_EVALUATION_REPORT,
    BASELINE_SENTENCE_SCORES,
    create_directories,
)


class BaselineTranslationEvaluator:
    """
    Evaluates baseline machine translation output using BLEU and chrF.

    Required input CSV columns:
        source_text
        target_text
        baseline_prediction
    """

    def __init__(
        self,
        prediction_file=BASELINE_TRANSLATION_FILE,
        reference_column="target_text",
        prediction_column="baseline_prediction",
        source_column="source_text",
    ):
        self.prediction_file = prediction_file
        self.reference_column = reference_column
        self.prediction_column = prediction_column
        self.source_column = source_column

        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)  # chrF++ style: character + word n-gram scoring

    def load_predictions(self) -> pd.DataFrame:
        if not self.prediction_file.exists():
            raise FileNotFoundError(
                f"\nPrediction file not found:\n{self.prediction_file}\n\n"
                "Reason: Step 3 baseline translation did not generate the output file.\n"
                "Please run run_baseline_translation.py successfully first.\n"
                "After that, run this evaluation script again."
            )

        df = pd.read_csv(self.prediction_file, encoding="utf-8-sig")

        required_columns = [
            self.source_column,
            self.reference_column,
            self.prediction_column,
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Detected columns: {list(df.columns)}\n\n"
                "Your baseline translation file must contain:\n"
                "source_text, target_text, baseline_prediction"
            )

        df = df.dropna(
            subset=[
                self.source_column,
                self.reference_column,
                self.prediction_column,
            ]
        )

        df[self.reference_column] = df[self.reference_column].astype(str).str.strip()
        df[self.prediction_column] = df[self.prediction_column].astype(str).str.strip()

        df = df[df[self.reference_column].str.len() > 0]
        df = df[df[self.prediction_column].str.len() > 0]

        if len(df) == 0:
            raise ValueError("No valid reference-prediction pairs found for evaluation.")

        return df.reset_index(drop=True)

    def corpus_evaluation(self, df: pd.DataFrame) -> dict:
        predictions = df[self.prediction_column].tolist()
        references = df[self.reference_column].tolist()

        bleu_score = self.bleu.corpus_score(
            predictions,
            [references],
        )

        chrf_score = self.chrf.corpus_score(
            predictions,
            [references],
        )

        report = {
            "total_evaluated_sentences": len(df),
            "metrics": {
                "BLEU": round(bleu_score.score, 4),
                "chrF++": round(chrf_score.score, 4),
            },
            "metric_signatures": {
                "BLEU": str(self.bleu.get_signature()),
                "chrF++": str(self.chrf.get_signature()),
            },
            "input_file": str(self.prediction_file),
            "reference_column": self.reference_column,
            "prediction_column": self.prediction_column,
        }

        return report

    def sentence_level_evaluation(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for index, row in df.iterrows():
            source = row[self.source_column]
            reference = row[self.reference_column]
            prediction = row[self.prediction_column]

            sent_bleu = self.bleu.sentence_score(
                prediction,
                [reference],
            ).score

            sent_chrf = self.chrf.sentence_score(
                prediction,
                [reference],
            ).score

            rows.append(
                {
                    "id": index + 1,
                    "source_text": source,
                    "reference_translation": reference,
                    "predicted_translation": prediction,
                    "sentence_bleu": round(sent_bleu, 4),
                    "sentence_chrf++": round(sent_chrf, 4),
                    "reference_length_chars": len(reference),
                    "prediction_length_chars": len(prediction),
                    "reference_length_words": len(reference.split()),
                    "prediction_length_words": len(prediction.split()),
                }
            )

        return pd.DataFrame(rows)

    def save_outputs(self, report: dict, sentence_scores_df: pd.DataFrame):
        BASELINE_EVALUATION_REPORT.parent.mkdir(parents=True, exist_ok=True)

        with open(BASELINE_EVALUATION_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            BASELINE_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

    def evaluate(self):
        create_directories()

        print("Loading baseline translation file...")
        df = self.load_predictions()

        print(f"Valid sentence pairs found: {len(df)}")

        print("Computing corpus-level BLEU and chrF++...")
        report = self.corpus_evaluation(df)

        print("Computing sentence-level scores...")
        sentence_scores_df = self.sentence_level_evaluation(df)

        print("Saving evaluation outputs...")
        self.save_outputs(report, sentence_scores_df)

        print("\nEvaluation completed successfully.")
        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nFiles saved:")
        print(f"Evaluation report: {BASELINE_EVALUATION_REPORT}")
        print(f"Sentence scores  : {BASELINE_SENTENCE_SCORES}")

        print("\nSample sentence-level scores:")
        print(sentence_scores_df.head(5).to_string(index=False))

        return report, sentence_scores_df


def run_baseline_evaluation():
    evaluator = BaselineTranslationEvaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    run_baseline_evaluation()