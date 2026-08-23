# src/train_finetune_nllb.py

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

from src.config import (
    TRAIN_FILE,
    VALID_FILE,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    FINETUNED_MODEL_DIR,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    create_directories,
)


def load_translation_dataset(csv_file):
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

    df = df[required_columns].dropna()
    df["source_text"] = df["source_text"].astype(str).str.strip()
    df["target_text"] = df["target_text"].astype(str).str.strip()

    df = df[df["source_text"].str.len() > 0]
    df = df[df["target_text"].str.len() > 0]

    return Dataset.from_pandas(df.reset_index(drop=True))


def train_finetuned_nllb():
    create_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Fine-tuning NLLB model")
    print(f"Device selected: {device}")

    if device == "cpu":
        print(
            "\nWARNING: CUDA GPU not detected. "
            "Training will run on CPU and may be slow.\n"
        )

    print(f"Base model: {BASELINE_MODEL_NAME}")
    print(f"Source language code: {SOURCE_LANG_CODE}")
    print(f"Target language code: {TARGET_LANG_CODE}")

    train_dataset = load_translation_dataset(TRAIN_FILE)
    valid_dataset = load_translation_dataset(VALID_FILE)

    print(f"Training records: {len(train_dataset)}")
    print(f"Validation records: {len(valid_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASELINE_MODEL_NAME,
        src_lang=SOURCE_LANG_CODE,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(BASELINE_MODEL_NAME)
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

    print("Tokenizing dataset...")

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

    use_fp16 = True if device == "cuda" else False

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(FINETUNED_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=TRAIN_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=5,
        fp16=use_fp16,
        logging_dir="outputs/training_logs",
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print("\nSaving fine-tuned model...")
    trainer.save_model(str(FINETUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINETUNED_MODEL_DIR))

    print("\nFine-tuning completed successfully.")
    print(f"Fine-tuned model saved at: {FINETUNED_MODEL_DIR}")


if __name__ == "__main__":
    train_finetuned_nllb()