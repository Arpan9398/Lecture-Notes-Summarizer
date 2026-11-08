"""Chunked abstractive summarization with a pretrained BART model."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .chunking import chunk_text
from .config import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_MAX_SUMMARY_TOKENS,
    DEFAULT_MIN_SUMMARY_TOKENS,
    DEFAULT_MODEL_NAME,
    MAX_INPUT_TOKENS,
    MAX_REDUCTION_PASSES,
    MIN_WORDS_TO_SUMMARIZE,
)
from .errors import EmptyTextError, TextTooShortError

ProgressCallback = Callable[[int, int], None]


@dataclass
class SummaryResult:
    summary: str
    chunk_summaries: List[str] = field(default_factory=list)
    num_chunks: int = 0
    input_words: int = 0
    summary_words: int = 0


class LectureSummarizer:
    """Loads a BART checkpoint once and summarizes arbitrarily long text."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers/torch are not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ core
    def _generate(self, text: str, max_tokens: int, min_tokens: int) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(self.device)
        n_input = int(inputs["input_ids"].shape[1])

        # Keep the summary shorter than its input and min < max.
        max_len = max(16, min(max_tokens, int(n_input * 0.8)))
        min_len = max(8, min(min_tokens, max_len - 8))

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=4,
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ---------------------------------------------------------------- public
    def summarize(
        self,
        text: str,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
        min_summary_tokens: int = DEFAULT_MIN_SUMMARY_TOKENS,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SummaryResult:
        """Summarize ``text``. Long inputs are chunked, summarized per chunk,
        then the chunk summaries are condensed into one final summary."""
        text = (text or "").strip()
        if not text:
            raise EmptyTextError("There is no text to summarize.")
        input_words = len(text.split())
        if input_words < MIN_WORDS_TO_SUMMARIZE:
            raise TextTooShortError(
                f"The text has only {input_words} words; at least "
                f"{MIN_WORDS_TO_SUMMARIZE} are needed for a meaningful summary."
            )

        chunks = chunk_text(text, max_words=chunk_words)
        multi = len(chunks) > 1
        total_steps = len(chunks) + (1 if multi else 0)
        done = 0

        def tick() -> None:
            nonlocal done
            done = min(done + 1, total_steps)
            if progress_callback:
                progress_callback(done, total_steps)

        chunk_summaries: List[str] = []
        for chunk in chunks:
            chunk_summaries.append(
                self._generate(chunk, max_summary_tokens, min_summary_tokens)
            )
            tick()

        if not multi:
            final = chunk_summaries[0]
        else:
            pieces = chunk_summaries
            for _ in range(MAX_REDUCTION_PASSES):
                if len(" ".join(pieces).split()) <= chunk_words:
                    break
                regrouped = chunk_text(" ".join(pieces), max_words=chunk_words)
                pieces = [
                    self._generate(c, max_summary_tokens, min_summary_tokens)
                    for c in regrouped
                ]
            combined = " ".join(pieces)
            if len(combined.split()) >= MIN_WORDS_TO_SUMMARIZE:
                final = self._generate(
                    combined, max_summary_tokens * 2, min_summary_tokens
                )
            else:
                final = combined
            tick()

        return SummaryResult(
            summary=final,
            chunk_summaries=chunk_summaries,
            num_chunks=len(chunks),
            input_words=input_words,
            summary_words=len(final.split()),
        )
