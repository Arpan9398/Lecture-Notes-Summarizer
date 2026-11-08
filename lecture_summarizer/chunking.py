"""Split long text into chunks that fit BART's input limit."""

import re
from typing import List

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])')


def split_sentences(text: str) -> List[str]:
    """Light-weight sentence splitter (no external downloads needed)."""
    sentences: List[str] = []
    for paragraph in re.split(r"\n\s*\n", (text or "").strip()):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip())
    return sentences


def _split_long_sentence(sentence: str, max_words: int) -> List[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_text(text: str, max_words: int = 500) -> List[str]:
    """Group whole sentences into chunks of at most ``max_words`` words.

    A single sentence longer than ``max_words`` is split by words.
    No text is dropped.
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")

    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for sentence in split_sentences(text):
        if len(sentence.split()) <= max_words:
            pieces = [sentence]
        else:
            pieces = _split_long_sentence(sentence, max_words)
        for piece in pieces:
            n = len(piece.split())
            if current and count + n > max_words:
                chunks.append(" ".join(current))
                current, count = [], 0
            current.append(piece)
            count += n
    if current:
        chunks.append(" ".join(current))
    return chunks
