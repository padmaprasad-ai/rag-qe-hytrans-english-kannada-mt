# src/generate_large_experiment_report.py

import json
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    LARGE_BASELINE_EVALUATION_REPORT,
    LARGE_LORA_EVALUATION_REPORT,
    LARGE_RAG_EVALUATION_REPORT,
    LARGE_RAG_THRESHOLD_TUNING_REPORT,
    LARGE_RAG_THRESHOLD_TUNING_CSV,
    LARGE_QA_EVALUATION_REPORT,
    TEST_VALIDATION_SELECTED_RAG_EVALUATION_REPORT,
    LARGE_CONSOLIDATED_RESULTS_CSV,
    LARGE_CONSOLIDATED_RESULTS_JSON,
    LARGE_MODEL_COMPARISON_CHART,
    LARGE_RAG_THRESHOLD_CHART,
    create_directories,
)


def load_json(path):
    if not path.exists():
        print(f"Warning: missing file: {path}")
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


class LargeExperimentReportGenerator:
    """
    Consolidates all large MT experiment results, including:
        1. Large baseline NLLB
        2. Large LoRA NLLB
        3. Fixed-threshold RAG
        4. Test-set threshold sensitivity results
        5. Validation-selected RAG-Hybrid
        6. Large quality-aware hybrid
    """

    def __init__(self):
        create_directories()

    def collect_results(self):
        rows = []

        baseline_report = load_json(LARGE_BASELINE_EVALUATION_REPORT)
        lora_report = load_json(LARGE_LORA_EVALUATION_REPORT)
        rag_report = load_json(LARGE_RAG_EVALUATION_REPORT)
        rag_tuning_report = load_json(LARGE_RAG_THRESHOLD_TUNING_REPORT)
        validation_selected_rag_report = load_json(
            TEST_VALIDATION_SELECTED_RAG_EVALUATION_REPORT
        )
        qa_report = load_json(LARGE_QA_EVALUATION_REPORT)

        if baseline_report:
            rows.append(
                {
                    "model": "Large Baseline NLLB",
                    "BLEU": baseline_report["metrics"]["BLEU"],
                    "chrF++": baseline_report["metrics"]["chrF++"],
                    "retrieval_selected": 0,
                    "nmt_lora_selected": baseline_report["total_evaluated_sentences"],
                    "result_type": "final_test_result",
                    "description": "Pretrained NLLB evaluated on large test set",
                }
            )

        if lora_report:
            rows.append(
                {
                    "model": "Large LoRA NLLB",
                    "BLEU": lora_report["metrics"]["BLEU"],
                    "chrF++": lora_report["metrics"]["chrF++"],
                    "retrieval_selected": 0,
                    "nmt_lora_selected": lora_report["total_evaluated_sentences"],
                    "result_type": "final_test_result",
                    "description": "LoRA-adapted NLLB using large parallel training split",
                }
            )

        if rag_report:
            method_counts = rag_report.get("method_selection_counts", {})
            rows.append(
                {
                    "model": "Large RAG-Hybrid Threshold 0.70",
                    "BLEU": rag_report["metrics"]["BLEU"],
                    "chrF++": rag_report["metrics"]["chrF++"],
                    "retrieval_selected": method_counts.get("retrieval_memory", 0),
                    "nmt_lora_selected": method_counts.get("large_lora_nllb", 0),
                    "result_type": "fixed_threshold_ablation",
                    "description": "RAG-Hybrid with fixed retrieval threshold 0.70",
                }
            )

        if rag_tuning_report:
            best_bleu = rag_tuning_report["best_by_bleu"]

            rows.append(
                {
                    "model": f"Best RAG by BLEU Threshold {best_bleu['threshold']}",
                    "BLEU": best_bleu["BLEU"],
                    "chrF++": best_bleu["chrF++"],
                    "retrieval_selected": best_bleu["retrieval_selected"],
                    "nmt_lora_selected": best_bleu["lora_selected"],
                    "result_type": "test_threshold_sensitivity_only",
                    "description": "Best threshold selected by BLEU during test-set sensitivity analysis; not used as final unbiased result",
                }
            )

            best_chrf = rag_tuning_report["best_by_chrf++"]

            rows.append(
                {
                    "model": f"Best RAG by chrF++ Threshold {best_chrf['threshold']}",
                    "BLEU": best_chrf["BLEU"],
                    "chrF++": best_chrf["chrF++"],
                    "retrieval_selected": best_chrf["retrieval_selected"],
                    "nmt_lora_selected": best_chrf["lora_selected"],
                    "result_type": "test_threshold_sensitivity_only",
                    "description": "Best threshold selected by chrF++ during test-set sensitivity analysis; not used as final unbiased result",
                }
            )

        if validation_selected_rag_report:
            method_counts = validation_selected_rag_report.get(
                "method_selection_counts",
                {},
            )

            rows.append(
                {
                    "model": (
                        "Validation-Selected RAG-Hybrid "
                        f"Threshold {validation_selected_rag_report['selected_threshold']}"
                    ),
                    "BLEU": validation_selected_rag_report["test_metrics"]["BLEU"],
                    "chrF++": validation_selected_rag_report["test_metrics"]["chrF++"],
                    "retrieval_selected": method_counts.get("retrieval_memory", 0),
                    "nmt_lora_selected": method_counts.get("large_lora_nllb", 0),
                    "result_type": "final_test_result",
                    "description": "RAG threshold selected on validation set and applied once to the test set",
                }
            )

        if qa_report:
            method_counts = qa_report.get("method_selection_counts", {})

            rows.append(
                {
                    "model": "Large Quality-Aware Hybrid",
                    "BLEU": qa_report["metrics"]["BLEU"],
                    "chrF++": qa_report["metrics"]["chrF++"],
                    "retrieval_selected": method_counts.get("retrieval_memory", 0),
                    "nmt_lora_selected": method_counts.get("large_lora_nllb", 0),
                    "result_type": "final_test_result",
                    "description": "Reference-free QE-based candidate selection between LoRA and retrieval",
                }
            )

        if not rows:
            raise ValueError(
                "No large experiment reports found. Please complete Steps 14–22 first."
            )

        return pd.DataFrame(rows)

    def save_results(self, results_df):
        results_df.to_csv(
            LARGE_CONSOLIDATED_RESULTS_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        with open(LARGE_CONSOLIDATED_RESULTS_JSON, "w", encoding="utf-8") as file:
            json.dump(
                results_df.to_dict(orient="records"),
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"Saved CSV : {LARGE_CONSOLIDATED_RESULTS_CSV}")
        print(f"Saved JSON: {LARGE_CONSOLIDATED_RESULTS_JSON}")

    def plot_model_comparison(self, results_df):
        final_df = results_df[
            results_df["result_type"].isin(
                [
                    "final_test_result",
                    "fixed_threshold_ablation",
                ]
            )
        ].copy()

        x_labels = final_df["model"].tolist()
        bleu_scores = final_df["BLEU"].tolist()
        chrf_scores = final_df["chrF++"].tolist()

        x_positions = range(len(x_labels))
        width = 0.35

        plt.figure(figsize=(15, 7))

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
        plt.title("Large-Corpus MT Model Comparison")
        plt.xticks(list(x_positions), x_labels, rotation=25, ha="right")
        plt.legend()
        plt.tight_layout()

        plt.savefig(LARGE_MODEL_COMPARISON_CHART, dpi=300)
        plt.close()

        print(f"Saved chart: {LARGE_MODEL_COMPARISON_CHART}")

    def plot_threshold_tuning(self):
        if not LARGE_RAG_THRESHOLD_TUNING_CSV.exists():
            print(
                f"Warning: threshold tuning CSV not found: "
                f"{LARGE_RAG_THRESHOLD_TUNING_CSV}"
            )
            return

        df = pd.read_csv(LARGE_RAG_THRESHOLD_TUNING_CSV, encoding="utf-8-sig")

        plt.figure(figsize=(11, 6))

        plt.plot(df["threshold"], df["BLEU"], marker="o", label="BLEU")
        plt.plot(df["threshold"], df["chrF++"], marker="s", label="chrF++")

        plt.xlabel("Retrieval Similarity Threshold")
        plt.ylabel("Score")
        plt.title("Large RAG Threshold Sensitivity Analysis")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(LARGE_RAG_THRESHOLD_CHART, dpi=300)
        plt.close()

        print(f"Saved threshold chart: {LARGE_RAG_THRESHOLD_CHART}")

    def print_summary(self, results_df):
        print("\n================ Large Consolidated MT Results ================")
        print(results_df.to_string(index=False))

        final_df = results_df[results_df["result_type"] == "final_test_result"].copy()

        best_final_bleu = final_df.sort_values(by="BLEU", ascending=False).iloc[0]
        best_final_chrf = final_df.sort_values(by="chrF++", ascending=False).iloc[0]

        print("\n================ Best Final Models ================")
        print(
            f"Best final BLEU model   : "
            f"{best_final_bleu['model']} | BLEU = {best_final_bleu['BLEU']}"
        )
        print(
            f"Best final chrF++ model : "
            f"{best_final_chrf['model']} | chrF++ = {best_final_chrf['chrF++']}"
        )

        print("\n================ Methodological Note ================")
        print(
            "Rows marked as test_threshold_sensitivity_only are useful for ablation "
            "analysis but should not be claimed as final unbiased model results. "
            "The validation-selected RAG-Hybrid row is the methodologically correct "
            "RAG result because its threshold was selected on the validation set and "
            "then applied once to the test set."
        )

        print("\n================ Research Interpretation ================")
        print(
            "The large-corpus experiments show that LoRA adaptation gives a marginal "
            "BLEU improvement over the pretrained baseline. Fixed low-threshold RAG "
            "degrades performance because many retrieval candidates are only loosely "
            "related to the input. Validation-based threshold selection chooses a very "
            "strict threshold, confirming that retrieval memory should be used only "
            "for near-exact or highly similar source sentences. The quality-aware hybrid "
            "selector behaves safely by rejecting most retrieval candidates, although "
            "its improvement over LoRA remains marginal."
        )

    def generate(self):
        results_df = self.collect_results()
        self.save_results(results_df)
        self.plot_model_comparison(results_df)
        self.plot_threshold_tuning()
        self.print_summary(results_df)

        print("\nLarge experiment report generation completed successfully.")

        return results_df


def run_large_experiment_report_generation():
    generator = LargeExperimentReportGenerator()
    return generator.generate()


if __name__ == "__main__":
    run_large_experiment_report_generation()