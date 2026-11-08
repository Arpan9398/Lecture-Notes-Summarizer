"""Central configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "reference" / "reference_summaries.json"
SAMPLE_PDF_PATH = PROJECT_ROOT / "data" / "sample" / "sample_lecture_notes.pdf"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "evaluation_results.json"

# Pretrained BART checkpoint fine-tuned for summarization.
DEFAULT_MODEL_NAME = "facebook/bart-large-cnn"

# BART accepts at most 1024 tokens; ~500 words stays safely below that.
MAX_INPUT_TOKENS = 1024
DEFAULT_CHUNK_WORDS = 500

# Texts shorter than this are rejected (too little to summarize).
MIN_WORDS_TO_SUMMARIZE = 30

# Summary length limits (in tokens) for each generated summary.
DEFAULT_MAX_SUMMARY_TOKENS = 130
DEFAULT_MIN_SUMMARY_TOKENS = 30

# If combined chunk summaries are still too long, summarize them again (max passes).
MAX_REDUCTION_PASSES = 3

# Characters of the original text shown in the UI preview.
PREVIEW_CHARS = 5000
