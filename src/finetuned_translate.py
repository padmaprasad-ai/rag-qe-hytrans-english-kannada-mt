# src/finetuned_translate.py

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import (
    TEST_FILE,
    FINETUNED_MODEL_DIR,
    FINETUNED_TRANSLATION_FILE,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    create_directories,
)


class FineTunedNLLBTranslator:
    """
    Generates translations using the fine-tuned NLLB model.
    """

    def __init__(self):
        if not FINETUNED_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found:\n{FINETUNED_MODEL_DIR}\n"
                "Please run Step 5 fine-tuning first."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading fine-tuned model...")
        print(f"Model path: {FINETUNED_MODEL_DIR}")
        print(f"Device selected: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(FINETUNED_MODEL_DIR),
            src_lang=SOURCE_LANG_CODE,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(FINETUNED_MODEL_DIR))
        self.model.to(self.device)
        self.model.eval()

        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            TARGET_LANG_CODE
        )

        if self.forced_bos_token_id is None:
            raise ValueError(f"Invalid target language code: {TARGET_LANG_CODE}")

    def translate_sentence(self, text: str) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        self.tokenizer.src_lang = SOURCE_LANG_CODE

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=MAX_TARGET_LENGTH,
                num_beams=5,
                early_stopping=True,
            )

        translation = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        return translation.strip()

    def translate_test_file(self, limit=None):
        create_directories()

        if not TEST_FILE.exists():
            raise FileNotFoundError(
                f"Test file not found:\n{TEST_FILE}\n"
                "Please run preprocessing first."
            )

        df = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

        if "source_text" not in df.columns:
            raise ValueError("Column 'source_text' not found in test file.")

        if limit is not None:
            df = df.head(limit).copy()

        predictions = []

        print(f"\nTranslating {len(df)} test sentences using fine-tuned model...")

        for source_text in tqdm(df["source_text"].tolist()):
            prediction = self.translate_sentence(source_text)
            predictions.append(prediction)

        df["finetuned_prediction"] = predictions
        df["finetuned_model_path"] = str(FINETUNED_MODEL_DIR)
        df["source_lang_code"] = SOURCE_LANG_CODE
        df["target_lang_code"] = TARGET_LANG_CODE

        df.to_csv(FINETUNED_TRANSLATION_FILE, index=False, encoding="utf-8-sig")

        print("\nFine-tuned translation completed successfully.")
        print(f"Output saved at: {FINETUNED_TRANSLATION_FILE}")

        print("\nSample outputs:")
        for _, row in df.head(5).iterrows():
            print("\n----------------------------------------")
            print(f"Source    : {row['source_text']}")

            if "target_text" in df.columns:
                print(f"Reference : {row['target_text']}")

            print(f"Prediction: {row['finetuned_prediction']}")

        return df


def run_finetuned_translation():
    translator = FineTunedNLLBTranslator()
    return translator.translate_test_file(limit=None)


if __name__ == "__main__":
    run_finetuned_translation()