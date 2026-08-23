# src/baseline_translate.py

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import (
    TEST_FILE,
    OUTPUT_DIR,
    BASELINE_MODEL_NAME,
    SOURCE_LANG_CODE,
    TARGET_LANG_CODE,
    BASELINE_TRANSLATION_FILE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    create_directories,
)


class NLLBBaselineTranslator:
    """
    Baseline translator using Facebook NLLB.

    This module performs zero-shot / pretrained translation before fine-tuning.
    It is used as the first MT baseline in the research framework.
    """

    def __init__(
        self,
        model_name: str = BASELINE_MODEL_NAME,
        source_lang: str = SOURCE_LANG_CODE,
        target_lang: str = TARGET_LANG_CODE,
    ):
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {self.model_name}")
        print(f"Device selected: {self.device}")
        print(f"Source language: {self.source_lang}")
        print(f"Target language: {self.target_lang}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.source_lang,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.target_lang
        )

        if self.forced_bos_token_id is None:
            raise ValueError(
                f"Invalid target language code: {self.target_lang}. "
                "Please check the NLLB language code."
            )

    def translate_sentence(self, text: str) -> str:
        """
        Translate a single sentence.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        self.tokenizer.src_lang = self.source_lang

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

    def translate_dataframe(
        self,
        input_file=TEST_FILE,
        output_file=BASELINE_TRANSLATION_FILE,
        limit: int = 10,
    ):
        """
        Translate sentences from test.csv and save output.
        limit is kept small initially to avoid long CPU execution.
        """
        create_directories()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not input_file.exists():
            raise FileNotFoundError(
                f"Test file not found: {input_file}\n"
                "Please run preprocessing first."
            )

        df = pd.read_csv(input_file, encoding="utf-8-sig")

        if "source_text" not in df.columns:
            raise ValueError("Column 'source_text' not found in test file.")

        if limit is not None:
            df = df.head(limit).copy()

        predictions = []

        print(f"\nTranslating {len(df)} sentences...")

        for source_text in tqdm(df["source_text"].tolist()):
            prediction = self.translate_sentence(source_text)
            predictions.append(prediction)

        df["baseline_prediction"] = predictions
        df["baseline_model"] = self.model_name
        df["source_lang_code"] = self.source_lang
        df["target_lang_code"] = self.target_lang

        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("\nBaseline translation completed successfully.")
        print(f"Output saved at: {output_file}")

        return df


def run_baseline_translation():
    translator = NLLBBaselineTranslator()

    result_df = translator.translate_dataframe(
        input_file=TEST_FILE,
        output_file=BASELINE_TRANSLATION_FILE,
        limit=10,
    )

    print("\nSample outputs:")

    for index, row in result_df.iterrows():
        print("\n----------------------------------------")
        print(f"Source: {row['source_text']}")

        if "target_text" in result_df.columns:
            print(f"Reference: {row['target_text']}")

        print(f"Prediction: {row['baseline_prediction']}")


if __name__ == "__main__":
    run_baseline_translation()