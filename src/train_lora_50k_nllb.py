# src/train_lora_50k_nllb.py

import json
import time

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import (
    TRAIN_FILE_50K,
    VALID_FILE_50K,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    LORA_MODEL_DIR_50K,
    LORA_ADAPTER_DIR_50K,
    LORA_TRAIN_EPOCHS_50K,
    LORA_TRAIN_BATCH_SIZE_50K,
    LORA_EVAL_BATCH_SIZE_50K,
    LORA_GRADIENT_ACCUMULATION_STEPS_50K,
    LORA_LEARNING_RATE_50K,
    LORA_MAX_TRAIN_SAMPLES_50K,
    LORA_MAX_VALID_SAMPLES_50K,
    LORA_R_50K,
    LORA_ALPHA_50K,
    LORA_DROPOUT_50K,
    LORA_TRAINING_LOG_DIR_50K,
    OUTPUT_DIR,
    create_directories,
)


# Local constants because the original experiment config did not define these separately.
LORA_WEIGHT_DECAY_50K = 0.01
LORA_SAVE_TOTAL_LIMIT_50K = 2
LORA_LOGGING_STEPS_50K = 50

LORA_TRAINING_REPORT_50K = OUTPUT_DIR / "lora_50k_training_report.json"


class TrainLoRA50K:
    """
    Step 29: LoRA fine-tuning of NLLB on the 50K English-Kannada corpus.

    Inputs:
        data/processed/train_50k.csv
        data/processed/valid_50k.csv

    Outputs:
        models/lora_50k_nllb_model/
        models/lora_50k_nllb_model/adapter/
        outputs/lora_50k_training_report.json
        outputs/lora_50k_training_logs/
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )

        self.tokenizer = None
        self.model = None

        self.train_dataset = None
        self.valid_dataset = None

        self.train_rows = 0
        self.valid_rows = 0

        self.trainable_params = 0
        self.total_params = 0
        self.trainable_percent = 0.0

    @staticmethod
    def load_dataframe(file_path, split_name):
        if not file_path.exists():
            raise FileNotFoundError(
                f"{split_name} file not found:\n{file_path}\n\n"
                "Please run Step 26 first: prepare_split_50k.py"
            )

        df = pd.read_csv(file_path, encoding="utf-8-sig")

        required_columns = ["source_text", "target_text"]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns in {split_name}: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=["source_text", "target_text"]).copy()

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()

        df = df[(df["source_text"] != "") & (df["target_text"] != "")]

        return df.reset_index(drop=True)

    def load_data(self):
        print("Loading 50K train and validation data...")

        train_df = self.load_dataframe(TRAIN_FILE_50K, "training")
        valid_df = self.load_dataframe(VALID_FILE_50K, "validation")

        if LORA_MAX_TRAIN_SAMPLES_50K is not None:
            train_df = train_df.head(LORA_MAX_TRAIN_SAMPLES_50K).copy()

        if LORA_MAX_VALID_SAMPLES_50K is not None:
            valid_df = valid_df.head(LORA_MAX_VALID_SAMPLES_50K).copy()

        self.train_rows = len(train_df)
        self.valid_rows = len(valid_df)

        print(f"Training records   : {self.train_rows}")
        print(f"Validation records : {self.valid_rows}")

        self.train_dataset = Dataset.from_pandas(
            train_df[["source_text", "target_text"]],
            preserve_index=False,
        )

        self.valid_dataset = Dataset.from_pandas(
            valid_df[["source_text", "target_text"]],
            preserve_index=False,
        )

    def load_model_and_tokenizer(self):
        print("\nLoading NLLB base model and tokenizer...")
        print(f"Base model : {BASELINE_MODEL_NAME}")
        print(f"Device     : {self.device}")
        print(f"GPU        : {self.gpu_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)
        self.tokenizer.src_lang = SOURCE_LANG_CODE

        self.model = AutoModelForSeq2SeqLM.from_pretrained(BASELINE_MODEL_NAME)

        lora_config = LoraConfig(
            r=LORA_R_50K,
            lora_alpha=LORA_ALPHA_50K,
            lora_dropout=LORA_DROPOUT_50K,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q_proj", "v_proj"],
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

        self.count_trainable_parameters()

        print("\nLoRA model initialized.")
        print(f"Trainable parameters : {self.trainable_params:,}")
        print(f"Total parameters     : {self.total_params:,}")
        print(f"Trainable %          : {self.trainable_percent:.4f}%")

    def count_trainable_parameters(self):
        trainable = 0
        total = 0

        for _, param in self.model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        self.trainable_params = trainable
        self.total_params = total
        self.trainable_percent = 100 * trainable / total

    def preprocess_batch(self, examples):
        self.tokenizer.src_lang = SOURCE_LANG_CODE

        model_inputs = self.tokenizer(
            examples["source_text"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=examples["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def tokenize_datasets(self):
        print("\nTokenizing train and validation datasets...")

        self.train_dataset = self.train_dataset.map(
            self.preprocess_batch,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing training split",
        )

        self.valid_dataset = self.valid_dataset.map(
            self.preprocess_batch,
            batched=True,
            remove_columns=self.valid_dataset.column_names,
            desc="Tokenizing validation split",
        )

        print("Tokenization completed.")

    def train(self):
        print("\nStarting 50K LoRA fine-tuning...")

        LORA_MODEL_DIR_50K.mkdir(parents=True, exist_ok=True)
        LORA_ADAPTER_DIR_50K.mkdir(parents=True, exist_ok=True)
        LORA_TRAINING_LOG_DIR_50K.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(LORA_MODEL_DIR_50K),
            num_train_epochs=LORA_TRAIN_EPOCHS_50K,
            per_device_train_batch_size=LORA_TRAIN_BATCH_SIZE_50K,
            per_device_eval_batch_size=LORA_EVAL_BATCH_SIZE_50K,
            gradient_accumulation_steps=LORA_GRADIENT_ACCUMULATION_STEPS_50K,
            learning_rate=LORA_LEARNING_RATE_50K,
            weight_decay=LORA_WEIGHT_DECAY_50K,
            logging_dir=str(LORA_TRAINING_LOG_DIR_50K),
            logging_steps=LORA_LOGGING_STEPS_50K,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=LORA_SAVE_TOTAL_LIMIT_50K,
            predict_with_generate=False,
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        start_time = time.time()

        train_result = trainer.train()
        eval_result = trainer.evaluate()

        runtime_seconds = round(time.time() - start_time, 2)

        print("\nSaving LoRA adapter and tokenizer...")

        self.model.save_pretrained(LORA_ADAPTER_DIR_50K)
        self.tokenizer.save_pretrained(LORA_MODEL_DIR_50K)

        report = {
            "step": "Step 29: 50K LoRA fine-tuning",
            "base_model": BASELINE_MODEL_NAME,
            "source_lang_code": SOURCE_LANG_CODE,
            "target_lang_code": TARGET_LANG_CODE,
            "train_file": str(TRAIN_FILE_50K),
            "valid_file": str(VALID_FILE_50K),
            "train_records": self.train_rows,
            "validation_records": self.valid_rows,
            "epochs": LORA_TRAIN_EPOCHS_50K,
            "train_batch_size": LORA_TRAIN_BATCH_SIZE_50K,
            "eval_batch_size": LORA_EVAL_BATCH_SIZE_50K,
            "gradient_accumulation_steps": LORA_GRADIENT_ACCUMULATION_STEPS_50K,
            "learning_rate": LORA_LEARNING_RATE_50K,
            "weight_decay": LORA_WEIGHT_DECAY_50K,
            "lora_rank": LORA_R_50K,
            "lora_alpha": LORA_ALPHA_50K,
            "lora_dropout": LORA_DROPOUT_50K,
            "trainable_parameters": self.trainable_params,
            "total_parameters": self.total_params,
            "trainable_percent": round(self.trainable_percent, 4),
            "device": self.device,
            "gpu": self.gpu_name,
            "runtime_seconds": runtime_seconds,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_result,
            "adapter_dir": str(LORA_ADAPTER_DIR_50K),
            "tokenizer_dir": str(LORA_MODEL_DIR_50K),
            "training_log_dir": str(LORA_TRAINING_LOG_DIR_50K),
        }

        with open(LORA_TRAINING_REPORT_50K, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        print("\nLoRA fine-tuning completed successfully.")
        print("\nSummary:")
        print(f"GPU                  : {self.gpu_name}")
        print(f"Training records     : {self.train_rows}")
        print(f"Validation records   : {self.valid_rows}")
        print(f"Epochs               : {LORA_TRAIN_EPOCHS_50K}")
        print(f"Trainable parameters : {self.trainable_params:,}")
        print(f"Trainable %          : {self.trainable_percent:.4f}%")
        print(f"Runtime seconds      : {runtime_seconds}")
        print(f"Adapter saved        : {LORA_ADAPTER_DIR_50K}")
        print(f"Tokenizer saved      : {LORA_MODEL_DIR_50K}")
        print(f"Report saved         : {LORA_TRAINING_REPORT_50K}")

    def run(self):
        create_directories()

        print("Step 29: 50K LoRA fine-tuning started.")

        self.load_data()
        self.load_model_and_tokenizer()
        self.tokenize_datasets()
        self.train()


def run_train_lora_50k_nllb():
    trainer = TrainLoRA50K()
    trainer.run()


if __name__ == "__main__":
    run_train_lora_50k_nllb()
