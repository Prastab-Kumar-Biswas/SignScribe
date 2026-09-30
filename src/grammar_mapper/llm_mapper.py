"""
M5b (inference) -- LLM-based gloss-to-sentence mapper (Path B).

Runs LOCALLY (CPU).  Loads a fine-tuned T5-small model (with LoRA
adapter weights from Colab) and generates fluent English sentences
from gloss token sequences.

The fine-tuned model and adapter weights are expected at:
    models/llm_finetuned/

If the fine-tuned model is not yet available, this module provides a
fallback using the base T5-small model (which won't be good at
gloss-to-sentence translation, but lets the demo run end-to-end
while you're still working on the Colab fine-tuning step).

Usage:
    from src.grammar_mapper.llm_mapper import LLMMapper
    mapper = LLMMapper()
    sentence = mapper.map("I SCHOOL GO")
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.config import (
    LLM_MODEL_NAME,
    LLM_FINETUNED_DIR,
    LLM_MAX_INPUT_LEN,
    LLM_MAX_OUTPUT_LEN,
)


class LLMMapper:
    """T5-small based gloss-to-English sentence mapper (Path B).

    Loads LoRA adapter weights on top of the base T5-small model
    for inference.  Falls back to base model if adapters are not found.
    """

    def __init__(
        self,
        model_dir: Path = LLM_FINETUNED_DIR,
        base_model_name: str = LLM_MODEL_NAME,
    ):
        """
        Parameters
        ----------
        model_dir : Path
            Directory containing the fine-tuned model / LoRA adapters.
        base_model_name : str
            Hugging Face model identifier for the base model.
        """
        self.device = torch.device("cpu")
        self._is_finetuned = False

        adapter_path = Path(model_dir)

        if adapter_path.exists() and any(adapter_path.iterdir()):
            # Try loading fine-tuned model (with PEFT adapters)
            try:
                from peft import PeftModel

                print(f"Loading base model: {base_model_name}")
                base_model = T5ForConditionalGeneration.from_pretrained(
                    base_model_name
                )
                print(f"Loading LoRA adapters from: {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    base_model, str(adapter_path)
                )
                self.model = self.model.merge_and_unload()
                self._is_finetuned = True
                print("Fine-tuned LLM loaded successfully.")
            except Exception as e:
                print(f"WARNING: Failed to load LoRA adapters: {e}")
                print("Falling back to base model.")
                self.model = T5ForConditionalGeneration.from_pretrained(
                    base_model_name
                )
        else:
            print(
                f"No fine-tuned model found at {adapter_path}. "
                f"Using base {base_model_name} (results will be poor "
                f"until you fine-tune on Colab)."
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                base_model_name
            )

        self.tokenizer = T5Tokenizer.from_pretrained(
            base_model_name, legacy=False
        )
        self.model.to(self.device)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"LLM parameters: {param_count:,}")
        print(f"Fine-tuned: {self._is_finetuned}")

    @property
    def is_finetuned(self) -> bool:
        """Whether the loaded model has fine-tuned weights."""
        return self._is_finetuned

    @torch.no_grad()
    def map(
        self,
        gloss_input: str | list[str],
        num_beams: int = 4,
        max_length: int = LLM_MAX_OUTPUT_LEN,
    ) -> str:
        """Generate an English sentence from gloss tokens.

        Parameters
        ----------
        gloss_input : str or list[str]
            Gloss tokens, e.g. "I SCHOOL GO" or ["I", "SCHOOL", "GO"].
        num_beams : int
            Beam search width (higher = better quality, slower).
        max_length : int
            Maximum output sequence length.

        Returns
        -------
        str
            Generated English sentence.
        """
        if isinstance(gloss_input, list):
            gloss_str = " ".join(gloss_input)
        else:
            gloss_str = gloss_input.strip()

        if not gloss_str:
            return ""

        # T5 expects a task prefix for seq2seq tasks
        input_text = f"translate gloss to english: {gloss_str}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=LLM_MAX_INPUT_LEN,
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

        sentence = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        return sentence

    def map_with_timing(
        self, gloss_input: str | list[str]
    ) -> tuple[str, float]:
        """Map glosses to English, also returning inference time in ms.

        Returns
        -------
        sentence : str
        inference_ms : float
        """
        t_start = time.perf_counter()
        sentence = self.map(gloss_input)
        t_end = time.perf_counter()
        return sentence, (t_end - t_start) * 1000.0

    def map_sequence(self, gloss_sequence: list[str]) -> str:
        """Map a sequence of gloss tokens, de-duplicating repeats.

        Same interface as RuleBasedMapper.map_sequence() for
        consistency in the demo.
        """
        deduped: list[str] = []
        for token in gloss_sequence:
            if not deduped or token != deduped[-1]:
                deduped.append(token)
        return self.map(deduped)


def demo():
    """Quick test of the LLM mapper."""
    mapper = LLMMapper()

    examples = [
        "I SCHOOL GO",
        "HE FOOD EAT",
        "SHE BOOK READ",
        "HELLO",
        "THANK YOU",
    ]

    print("\n" + "=" * 60)
    print("  SignScribe - LLM Mapper Demo (Path B)")
    print("=" * 60)
    print(f"  Fine-tuned: {mapper.is_finetuned}")
    print()

    for gloss in examples:
        sentence, ms = mapper.map_with_timing(gloss)
        print(f"  Gloss   : {gloss}")
        print(f"  English : {sentence}")
        print(f"  Time    : {ms:.0f} ms")
        print()

    print("=" * 60)


if __name__ == "__main__":
    demo()
