# src/config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

RAW_PARALLEL_FILE = RAW_DATA_DIR / "parallel.csv"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VALID_FILE = PROCESSED_DATA_DIR / "valid.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

NMT_MODEL_DIR = MODEL_DIR / "nmt_model"
RETRIEVER_DIR = MODEL_DIR / "retriever"

FAISS_INDEX_FILE = RETRIEVER_DIR / "tm_faiss.index"
MEMORY_FILE = RETRIEVER_DIR / "tm_memory.pkl"

TRANSLATION_OUTPUT_FILE = OUTPUT_DIR / "translations.csv"

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
RANDOM_SEED = 42


def create_directories():
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        NMT_MODEL_DIR,
        RETRIEVER_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Translation model settings

BASELINE_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB language codes
SOURCE_LANG_CODE = "eng_Latn"
TARGET_LANG_CODE = "kan_Knda"

BASELINE_TRANSLATION_FILE = OUTPUT_DIR / "baseline_translations.csv"

BASELINE_EVALUATION_REPORT = OUTPUT_DIR / "baseline_evaluation_report.json"
BASELINE_SENTENCE_SCORES = OUTPUT_DIR / "baseline_sentence_scores.csv"

# Fine-tuning settings

FINETUNED_MODEL_DIR = MODEL_DIR / "finetuned_nllb_model"
FINETUNED_TRANSLATION_FILE = OUTPUT_DIR / "finetuned_translations.csv"

TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 3e-5

# Fine-tuned model evaluation and comparison files

FINETUNED_EVALUATION_REPORT = OUTPUT_DIR / "finetuned_evaluation_report.json"
FINETUNED_SENTENCE_SCORES = OUTPUT_DIR / "finetuned_sentence_scores.csv"

MODEL_COMPARISON_REPORT = OUTPUT_DIR / "baseline_vs_finetuned_report.json"
MODEL_COMPARISON_SENTENCE_SCORES = OUTPUT_DIR / "baseline_vs_finetuned_sentence_scores.csv"

# RAG hybrid translation settings

HYBRID_TRANSLATION_FILE = OUTPUT_DIR / "hybrid_rag_translations.csv"
HYBRID_EVALUATION_REPORT = OUTPUT_DIR / "hybrid_rag_evaluation_report.json"
HYBRID_SENTENCE_SCORES = OUTPUT_DIR / "hybrid_rag_sentence_scores.csv"

RAG_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.88

# RAG threshold tuning files

RAG_THRESHOLD_TUNING_REPORT = OUTPUT_DIR / "rag_threshold_tuning_report.json"
RAG_THRESHOLD_TUNING_CSV = OUTPUT_DIR / "rag_threshold_tuning_results.csv"

# Quality-aware hybrid translation files

QUALITY_AWARE_TRANSLATION_FILE = OUTPUT_DIR / "quality_aware_hybrid_translations.csv"
QUALITY_AWARE_EVALUATION_REPORT = OUTPUT_DIR / "quality_aware_hybrid_evaluation_report.json"
QUALITY_AWARE_SENTENCE_SCORES = OUTPUT_DIR / "quality_aware_hybrid_sentence_scores.csv"

QE_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Consolidated experiment report files

CONSOLIDATED_RESULTS_CSV = OUTPUT_DIR / "consolidated_experiment_results.csv"
CONSOLIDATED_RESULTS_JSON = OUTPUT_DIR / "consolidated_experiment_results.json"
BLEU_CHRF_COMPARISON_CHART = OUTPUT_DIR / "bleu_chrf_model_comparison.png"
RAG_THRESHOLD_CHART = OUTPUT_DIR / "rag_threshold_tuning_chart.png"

# Online dataset download settings

HF_DATASET_NAME = "ai4bharat/samanantar"
HF_DATASET_CONFIG = "kn"
HF_DATASET_SPLIT = "train"

HF_SOURCE_LANG = "en"
HF_TARGET_LANG = "kn"

HF_MAX_SAMPLES = 5000
HF_OUTPUT_FILE = RAW_DATA_DIR / "large_parallel.csv"

# Large parallel corpus settings

LARGE_RAW_PARALLEL_FILE = RAW_DATA_DIR / "large_parallel.csv"
LARGE_CLEAN_PARALLEL_FILE = PROCESSED_DATA_DIR / "large_parallel_clean.csv"
LARGE_DATA_PREPARATION_LOG = OUTPUT_DIR / "large_data_preparation_log.json"

LARGE_DATA_CHUNK_SIZE = 50000

LARGE_MIN_SOURCE_CHARS = 2
LARGE_MIN_TARGET_CHARS = 2
LARGE_MAX_SOURCE_CHARS = 500
LARGE_MAX_TARGET_CHARS = 500

LARGE_MAX_LENGTH_RATIO = 3.0

# Large corpus advanced filtering and split files

LARGE_FILTERED_PARALLEL_FILE = PROCESSED_DATA_DIR / "large_parallel_filtered.csv"

TRAIN_LARGE_FILE = PROCESSED_DATA_DIR / "train_large.csv"
VALID_LARGE_FILE = PROCESSED_DATA_DIR / "valid_large.csv"
TEST_LARGE_FILE = PROCESSED_DATA_DIR / "test_large.csv"

LARGE_SPLIT_LOG = OUTPUT_DIR / "large_split_log.json"

LARGE_TRAIN_RATIO = 0.80
LARGE_VALID_RATIO = 0.10
LARGE_TEST_RATIO = 0.10

LARGE_MIN_SOURCE_WORDS = 2
LARGE_MIN_TARGET_WORDS = 1

LARGE_MAX_SOURCE_WORDS = 100
LARGE_MAX_TARGET_WORDS = 120

LARGE_ADVANCED_MAX_LENGTH_RATIO = 2.8

# Kannada Unicode range. For another target language, this can be changed later.
KANNADA_SCRIPT_REGEX = r"[\u0C80-\u0CFF]"
ENABLE_KANNADA_SCRIPT_FILTER = True

# Large retriever settings

LARGE_RETRIEVER_DIR = MODEL_DIR / "large_retriever"

LARGE_FAISS_INDEX_FILE = LARGE_RETRIEVER_DIR / "large_tm_faiss.index"
LARGE_MEMORY_FILE = LARGE_RETRIEVER_DIR / "large_tm_memory.pkl"
LARGE_SKLEARN_INDEX_FILE = LARGE_RETRIEVER_DIR / "large_sklearn_retriever.pkl"

LARGE_RETRIEVER_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LARGE_RETRIEVER_BATCH_SIZE = 64

# Keep None to use all training rows.
# For very large future experiments, set 100000 or 200000.
LARGE_RETRIEVER_MAX_ROWS = None

# Large baseline translation and evaluation files

LARGE_BASELINE_TRANSLATION_FILE = OUTPUT_DIR / "large_baseline_translations.csv"
LARGE_BASELINE_EVALUATION_REPORT = OUTPUT_DIR / "large_baseline_evaluation_report.json"
LARGE_BASELINE_SENTENCE_SCORES = OUTPUT_DIR / "large_baseline_sentence_scores.csv"

LARGE_TRANSLATION_BATCH_SIZE = 4

# Use None for full test set. For quick debugging, set 20 or 50.
LARGE_TEST_TRANSLATION_LIMIT = None

# Large LoRA fine-tuning settings

LARGE_LORA_MODEL_DIR = MODEL_DIR / "large_lora_nllb_model"
LARGE_LORA_ADAPTER_DIR = LARGE_LORA_MODEL_DIR / "adapter"

LARGE_LORA_TRAIN_EPOCHS = 2
LARGE_LORA_TRAIN_BATCH_SIZE = 1
LARGE_LORA_EVAL_BATCH_SIZE = 1
LARGE_LORA_GRADIENT_ACCUMULATION_STEPS = 8
LARGE_LORA_LEARNING_RATE = 2e-4

# For first safe run, keep 1000.
# Later set None to use all 3945 training rows.
LARGE_LORA_MAX_TRAIN_SAMPLES = None
LARGE_LORA_MAX_VALID_SAMPLES = None

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Large LoRA translation and evaluation files

LARGE_LORA_TRANSLATION_FILE = OUTPUT_DIR / "large_lora_translations.csv"
LARGE_LORA_EVALUATION_REPORT = OUTPUT_DIR / "large_lora_evaluation_report.json"
LARGE_LORA_SENTENCE_SCORES = OUTPUT_DIR / "large_lora_sentence_scores.csv"

LARGE_MODEL_COMPARISON_REPORT = OUTPUT_DIR / "large_baseline_vs_lora_report.json"
LARGE_MODEL_COMPARISON_SENTENCE_SCORES = OUTPUT_DIR / "large_baseline_vs_lora_sentence_scores.csv"

LARGE_LORA_TRANSLATION_BATCH_SIZE = 2

# Use None for full test set. Use 20 for quick testing.
LARGE_LORA_TEST_TRANSLATION_LIMIT = None

# Large RAG-hybrid translation and evaluation files

LARGE_RAG_TRANSLATION_FILE = OUTPUT_DIR / "large_rag_hybrid_translations.csv"
LARGE_RAG_EVALUATION_REPORT = OUTPUT_DIR / "large_rag_hybrid_evaluation_report.json"
LARGE_RAG_SENTENCE_SCORES = OUTPUT_DIR / "large_rag_hybrid_sentence_scores.csv"

LARGE_RAG_TOP_K = 3

# Start with a strict threshold. Later we tune it.
LARGE_RAG_SIMILARITY_THRESHOLD = 0.70

# Large RAG threshold tuning files

LARGE_RAG_THRESHOLD_TUNING_CSV = OUTPUT_DIR / "large_rag_threshold_tuning_results.csv"
LARGE_RAG_THRESHOLD_TUNING_REPORT = OUTPUT_DIR / "large_rag_threshold_tuning_report.json"


# Large quality-aware hybrid files

LARGE_QA_TRANSLATION_FILE = OUTPUT_DIR / "large_quality_aware_hybrid_translations.csv"
LARGE_QA_EVALUATION_REPORT = OUTPUT_DIR / "large_quality_aware_hybrid_evaluation_report.json"
LARGE_QA_SENTENCE_SCORES = OUTPUT_DIR / "large_quality_aware_hybrid_sentence_scores.csv"

LARGE_QA_COMPARISON_REPORT = OUTPUT_DIR / "large_quality_aware_comparison_report.json"

LARGE_QE_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Retrieval candidates below this similarity are strongly discouraged.
LARGE_QA_MIN_RETRIEVAL_SIMILARITY = 0.85

# Large consolidated experiment report files

LARGE_CONSOLIDATED_RESULTS_CSV = OUTPUT_DIR / "large_consolidated_experiment_results.csv"
LARGE_CONSOLIDATED_RESULTS_JSON = OUTPUT_DIR / "large_consolidated_experiment_results.json"

LARGE_MODEL_COMPARISON_CHART = OUTPUT_DIR / "large_model_comparison_chart.png"
LARGE_RAG_THRESHOLD_CHART = OUTPUT_DIR / "large_rag_threshold_chart.png"

# Validation-based RAG threshold selection

VALIDATION_RAG_CANDIDATES_FILE = OUTPUT_DIR / "validation_rag_candidates.csv"
VALIDATION_RAG_THRESHOLD_TUNING_CSV = OUTPUT_DIR / "validation_rag_threshold_tuning_results.csv"
VALIDATION_RAG_THRESHOLD_SELECTION_REPORT = OUTPUT_DIR / "validation_rag_threshold_selection_report.json"

TEST_VALIDATION_SELECTED_RAG_TRANSLATION_FILE = OUTPUT_DIR / "test_validation_selected_rag_translations.csv"
TEST_VALIDATION_SELECTED_RAG_EVALUATION_REPORT = OUTPUT_DIR / "test_validation_selected_rag_evaluation_report.json"
TEST_VALIDATION_SELECTED_RAG_SENTENCE_SCORES = OUTPUT_DIR / "test_validation_selected_rag_sentence_scores.csv"

VALIDATION_RAG_TRANSLATION_BATCH_SIZE = 2

# Use None for full validation set. Use 50 only for debugging.
VALIDATION_RAG_TRANSLATION_LIMIT = None

VALIDATION_RAG_TOP_K = 3

# Reuse validation candidates if the file already exists.
REUSE_VALIDATION_RAG_CANDIDATES = True

VALIDATION_RAG_THRESHOLDS = [
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

# ==============================
# 50K Samanantar experiment files
# ==============================

HF_DATASET_NAME_50K = "ai4bharat/samanantar"
HF_DATASET_CONFIG_50K = "kn"
HF_DATASET_SPLIT_50K = "train"

HF_MAX_SAMPLES_50K = 50000

HF_OUTPUT_FILE_50K = RAW_DATA_DIR / "parallel_50k.csv"
HF_SOURCE_ONLY_FILE_50K = RAW_DATA_DIR / "source_sentences_50k.txt"
HF_50K_DOWNLOAD_LOG = OUTPUT_DIR / "hf_50k_download_log.json"

# ==============================
# 50K cleaning and split settings
# ==============================

RAW_PARALLEL_FILE_50K = HF_OUTPUT_FILE_50K

CLEAN_PARALLEL_FILE_50K = PROCESSED_DATA_DIR / "parallel_50k_clean.csv"
TRAIN_FILE_50K = PROCESSED_DATA_DIR / "train_50k.csv"
VALID_FILE_50K = PROCESSED_DATA_DIR / "valid_50k.csv"
TEST_FILE_50K = PROCESSED_DATA_DIR / "test_50k.csv"

PREPARE_50K_LOG = OUTPUT_DIR / "prepare_50k_log.json"

CHUNK_SIZE_50K = 50000

TRAIN_RATIO_50K = 0.80
VALID_RATIO_50K = 0.10
TEST_RATIO_50K = 0.10

MIN_SOURCE_CHARS_50K = 2
MIN_TARGET_CHARS_50K = 2
MAX_SOURCE_CHARS_50K = 500
MAX_TARGET_CHARS_50K = 500

MIN_SOURCE_WORDS_50K = 2
MIN_TARGET_WORDS_50K = 1
MAX_SOURCE_WORDS_50K = 100
MAX_TARGET_WORDS_50K = 120

MAX_LENGTH_RATIO_50K = 3.0

KANNADA_SCRIPT_REGEX_50K = r"[\u0C80-\u0CFF]"
ENABLE_KANNADA_SCRIPT_FILTER_50K = True

# ==============================
# 50K FAISS retriever settings
# ==============================

RETRIEVER_DIR_50K = MODEL_DIR / "retriever_50k"

FAISS_INDEX_FILE_50K = RETRIEVER_DIR_50K / "tm_50k_faiss.index"
MEMORY_FILE_50K = RETRIEVER_DIR_50K / "tm_50k_memory.pkl"
SKLEARN_INDEX_FILE_50K = RETRIEVER_DIR_50K / "tm_50k_sklearn_retriever.pkl"

RETRIEVER_EMBEDDING_MODEL_50K = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVER_BATCH_SIZE_50K = 128

# Keep None to use all 39,642 training rows.
RETRIEVER_MAX_ROWS_50K = None

# ==============================
# 50K baseline NLLB evaluation
# ==============================

BASELINE_TRANSLATION_FILE_50K = OUTPUT_DIR / "baseline_50k_translations.csv"
BASELINE_EVALUATION_REPORT_50K = OUTPUT_DIR / "baseline_50k_evaluation_report.json"
BASELINE_SENTENCE_SCORES_50K = OUTPUT_DIR / "baseline_50k_sentence_scores.csv"

BASELINE_TRANSLATION_BATCH_SIZE_50K = 4

# For debugging, use 100.
# For final experiment, set None.
BASELINE_TEST_LIMIT_50K = None

# Saves output after every batch, useful for long CPU runs.
SAVE_EVERY_BATCH_50K = True

# ==============================
# 50K LoRA fine-tuning settings
# ==============================

LORA_MODEL_DIR_50K = MODEL_DIR / "lora_50k_nllb_model"
LORA_ADAPTER_DIR_50K = LORA_MODEL_DIR_50K / "adapter"

LORA_TRAIN_EPOCHS_50K = 2
LORA_TRAIN_BATCH_SIZE_50K = 1
LORA_EVAL_BATCH_SIZE_50K = 1
LORA_GRADIENT_ACCUMULATION_STEPS_50K = 8
LORA_LEARNING_RATE_50K = 2e-4

# Debug run: 5000 or 10000.
# Final paper run: None.
LORA_MAX_TRAIN_SAMPLES_50K = None

# Keep validation small during training to reduce time.
# Final evaluation will be done separately on the full test set.
LORA_MAX_VALID_SAMPLES_50K = 1000

LORA_R_50K = 8
LORA_ALPHA_50K = 16
LORA_DROPOUT_50K = 0.05

LORA_TRAINING_LOG_DIR_50K = OUTPUT_DIR / "lora_50k_training_logs"
