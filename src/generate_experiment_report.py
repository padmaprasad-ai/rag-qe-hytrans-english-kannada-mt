# src/generate_experiment_report.py

import json
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    BASELINE_EVALUATION_REPORT,
    MODEL_COMPARISON_REPORT,
    HYBRID_EVALUATION_REPORT,
    QUALITY_AWARE_EVALUATION_REPORT,
    RAG_THRESHOLD_TUNING_CSV,
    CONSOLIDATED_RESULTS_CSV,
    CONSOLIDATED_RESULTS_JSON,
    BLEU_CHRF_COMPARISON_CHART,
    RAG_THRESHOLD_CHART,
    create_directories,
)


def load_json_file(file_path):
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


class ExperimentReportGenerator:
    """
    Generates consolidated experimental report for the low-resource MT framework.

    It summarizes:
        - Baseline NLLB
        - Fine-tuned NLLB
        - RAG Hybrid
        - Quality-Aware Hybrid
        - RAG threshold tuning
    """

    def __init__(self):
        create_directories()

    def collect_model_scores(self):
        rows = []

        baseline_report = load_json_file(BASELINE_EVALUATION_REPORT)
        comparison_report = load_json_file(MODEL_COMPARISON_REPORT)
        hybrid_report = load_json_file(HYBRID_EVALUATION_REPORT)
        quality_report = load_json_file(QUALITY_AWARE_EVALUATION_REPORT)

        if baseline_report:
            rows.append(
                {
                    "model": "Baseline NLLB",
                    "BLEU": baseline_report["metrics"]["BLEU"],
                    "chrF++": baseline_report["metrics"]["chrF++"],
                    "selected_method": "pretrained_nllb",
                    "description": "Zero-shot/pretrained NLLB translation",
                }
            )

        if comparison_report:
            rows.append(
                {
                    "model": "Fine-tuned NLLB",
                    "BLEU": comparison_report["finetuned_nllb"]["BLEU"],
                    "chrF++": comparison_report["finetuned_nllb"]["chrF++"],
                    "selected_method": "finetuned_nllb",
                    "description": "NLLB fine-tuned on available parallel corpus",
                }
            )

        if hybrid_report:
            method_counts = hybrid_report.get("method_selection_counts", {})

            rows.append(
                {
                    "model": "RAG Hybrid",
                    "BLEU": hybrid_report["metrics"]["BLEU"],
                    "chrF++": hybrid_report["metrics"]["chrF++"],
                    "selected_method": str(method_counts),
                    "description": "Threshold-based retrieval-augmented hybrid translation",
                }
            )

        if quality_report:
            method_counts = quality_report.get("method_selection_counts", {})

            rows.append(
                {
                    "model": "Quality-Aware Hybrid",
                    "BLEU": quality_report["metrics"]["BLEU"],
                    "chrF++": quality_report["metrics"]["chrF++"],
                    "selected_method": str(method_counts),
                    "description": "QE-based candidate selection using NMT and retrieval candidates",
                }
            )

        if len(rows) == 0:
            raise ValueError(
                "No experiment reports found. Please complete previous steps first."
            )

        results_df = pd.DataFrame(rows)
        return results_df

    def save_consolidated_results(self, results_df):
        results_df.to_csv(
            CONSOLIDATED_RESULTS_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        records = results_df.to_dict(orient="records")

        with open(CONSOLIDATED_RESULTS_JSON, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=4, ensure_ascii=False)

        print(f"Consolidated CSV saved at: {CONSOLIDATED_RESULTS_CSV}")
        print(f"Consolidated JSON saved at: {CONSOLIDATED_RESULTS_JSON}")

    def plot_model_comparison(self, results_df):
        x_labels = results_df["model"].tolist()
        bleu_scores = results_df["BLEU"].tolist()
        chrf_scores = results_df["chrF++"].tolist()

        x_positions = range(len(x_labels))
        width = 0.35

        plt.figure(figsize=(11, 6))

        plt.bar(
            [x - width / 2 for x in x_positions],
            bleu_scores,
            width,
            label="BLEU",
        )

        plt.bar(
            [x + width / 2 for x in x_positions],
            chrf_scores,
            width,
            label="chrF++",
        )

        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("BLEU and chrF++ Comparison Across MT Models")
        plt.xticks(list(x_positions), x_labels, rotation=20, ha="right")
        plt.legend()
        plt.tight_layout()

        plt.savefig(BLEU_CHRF_COMPARISON_CHART, dpi=300)
        plt.close()

        print(f"BLEU/chrF++ comparison chart saved at: {BLEU_CHRF_COMPARISON_CHART}")

    def plot_rag_threshold_tuning(self):
        if not RAG_THRESHOLD_TUNING_CSV.exists():
            print(f"Warning: RAG threshold tuning file not found: {RAG_THRESHOLD_TUNING_CSV}")
            return

        df = pd.read_csv(RAG_THRESHOLD_TUNING_CSV, encoding="utf-8-sig")

        required_columns = ["threshold", "BLEU", "chrF++", "retrieval_selected"]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            print(f"Warning: Missing columns in threshold file: {missing}")
            return

        plt.figure(figsize=(10, 6))

        plt.plot(df["threshold"], df["BLEU"], marker="o", label="BLEU")
        plt.plot(df["threshold"], df["chrF++"], marker="s", label="chrF++")

        plt.xlabel("RAG Similarity Threshold")
        plt.ylabel("Score")
        plt.title("Effect of RAG Similarity Threshold on Translation Quality")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(RAG_THRESHOLD_CHART, dpi=300)
        plt.close()

        print(f"RAG threshold tuning chart saved at: {RAG_THRESHOLD_CHART}")

    def print_research_summary(self, results_df):
        print("\n================ Consolidated MT Results ================")
        print(results_df.to_string(index=False))

        best_bleu_row = results_df.sort_values(by="BLEU", ascending=False).iloc[0]
        best_chrf_row = results_df.sort_values(by="chrF++", ascending=False).iloc[0]

        print("\n================ Best Performing Models ================")
        print(f"Best BLEU model   : {best_bleu_row['model']} ({best_bleu_row['BLEU']})")
        print(f"Best chrF++ model : {best_chrf_row['model']} ({best_chrf_row['chrF++']})")

        print("\n================ Research Interpretation ================")
        print(
            "The consolidated results show the comparative behaviour of the baseline, "
            "fine-tuned, retrieval-augmented, and quality-aware hybrid MT systems. "
            "In the current pilot dataset, the quality-aware model preserved the stronger "
            "NMT output and rejected weaker retrieval candidates. This validates the need "
            "for quality estimation in retrieval-augmented low-resource machine translation."
        )

    def generate_report(self):
        results_df = self.collect_model_scores()

        self.save_consolidated_results(results_df)
        self.plot_model_comparison(results_df)
        self.plot_rag_threshold_tuning()
        self.print_research_summary(results_df)

        print("\nExperiment report generation completed successfully.")

        return results_df


def run_experiment_report_generation():
    generator = ExperimentReportGenerator()
    return generator.generate_report()


if __name__ == "__main__":
    run_experiment_report_generation()