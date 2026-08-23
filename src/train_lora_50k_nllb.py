# src/train_large_lora_nllb.py

import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType

from src.config import (
    TRAIN_LARGE_FILE,
    VALID_LARGE_FILE,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    LARGE_LORA_MODEL_DIR,
    LARGE_LORA_ADAPTER_DIR,
    LARGE_LORA_TRAIN_EPOCHS,
    LARGE_LORA_TRAIN_BATCH_SIZE,
    LARGE_LORA_EVAL_BATCH_SIZE,
    LARGE_LORA_GRADIENT_ACCUMULATION_STEPS,
    LARGE_LORA_LEARNING_RATE,
    LARGE_LORA_MAX_TRAIN_SAMPLES,
    LARGE_LORA_MAX_VALID_SAMPLES,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    create_directories,
)


def load_large_dataset(csv_file, max_samples=None):
    if not csv_file.exists():
        raise FileNotFoundError(f"File not found: {csv_file}")

    df = pd.read_csv(csv_file, encoding="utf-8-sig")

    required_columns = ["source_text", "target_text"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Required columns are source_text and target_text."
        )

    df = df[required_columns].dropna().reset_index(drop=True)

    df["source_text"] = df["source_text"].astype(str).str.strip()
    df["target_text"] = df["target_text"].astype(str).str.strip()

    df = df[df["source_text"].str.len() > 0]
    df = df[df["target_text"].str.len() > 0]

    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    return Dataset.from_pandas(df.reset_index(drop=True))


def apply_lora_to_model(model):
    """
    Applies LoRA adapters to the NLLB seq2seq model.
    """

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    print("\nLoRA configuration applied.")
    model.print_trainable_parameters()

    return model


def build_training_args(device):
    """
    Handles version differences in Transformers:
    some versions use eval_strategy, older versions use evaluation_strategy.
    """

    common_args = dict(
        output_dir=str(LARGE_LORA_MODEL_DIR),
        save_strategy="epoch",
        learning_rate=LARGE_LORA_LEARNING_RATE,
        per_device_train_batch_size=LARGE_LORA_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=LARGE_LORA_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=LARGE_LORA_GRADIENT_ACCUMULATION_STEPS,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=LARGE_LORA_TRAIN_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        fp16=True if device == "cuda" else False,
        logging_dir="outputs/large_lora_training_logs",
        logging_steps=25,
        report_to="none",
    )

    try:
        return Seq2SeqTrainingArguments(
            eval_strategy="epoch",
            **common_args,
        )
    except TypeError:
        return Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            **common_args,
        )


def train_large_lora_nllb():
    create_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Step 15: Large-data LoRA fine-tuning started.")
    print(f"Base model      : {BASELINE_MODEL_NAME}")
    print(f"Device selected : {device}")
    print(f"Source language : {SOURCE_LANG_CODE}")
    print(f"Target language : {TARGET_LANG_CODE}")

    if device == "cpu":
        print(
            "\nWARNING: CUDA GPU not detected. "
            "LoRA training will still work, but it may take considerable time on CPU.\n"
        )

    train_dataset = load_large_dataset(
        TRAIN_LARGE_FILE,
        max_samples=LARGE_LORA_MAX_TRAIN_SAMPLES,
    )

    valid_dataset = load_large_dataset(
        VALID_LARGE_FILE,
        max_samples=LARGE_LORA_MAX_VALID_SAMPLES,
    )

    print(f"Training records used   : {len(train_dataset)}")
    print(f"Validation records used : {len(valid_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASELINE_MODEL_NAME,
        src_lang=SOURCE_LANG_CODE,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(BASELINE_MODEL_NAME)
    model = apply_lora_to_model(model)
    model.to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG_CODE)

    if forced_bos_token_id is None:
        raise ValueError(f"Invalid target language code: {TARGET_LANG_CODE}")

    def preprocess_function(batch):
        tokenizer.src_lang = SOURCE_LANG_CODE

        model_inputs = tokenizer(
            batch["source_text"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing large training dataset...")

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_valid = valid_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=valid_dataset.column_names,
    )

    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
        )

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = sacrebleu.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        return {"bleu": round(result["score"], 4)}

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = build_training_args(device)

    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    print("\nStarting large LoRA fine-tuning...")
    trainer.train()

    print("\nSaving LoRA adapter and tokenizer...")

    LARGE_LORA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LARGE_LORA_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(LARGE_LORA_ADAPTER_DIR))
    tokenizer.save_pretrained(str(LARGE_LORA_MODEL_DIR))

    print("\nLarge-data LoRA fine-tuning completed successfully.")
    print(f"LoRA adapter saved at : {LARGE_LORA_ADAPTER_DIR}")
    print(f"Tokenizer saved at    : {LARGE_LORA_MODEL_DIR}")


if __name__ == "__main__":
    train_large_lora_nllb()