"""Command-line ROUGE evaluation on the bundled reference dataset.

Usage:
    python evaluate.py
    python evaluate.py --limit 2 --output results/my_run.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from lecture_summarizer.config import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_DATASET_PATH,
    DEFAULT_MAX_SUMMARY_TOKENS,
    DEFAULT_MIN_SUMMARY_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RESULTS_PATH,
)
from lecture_summarizer.errors import LectureSummarizerError
from lecture_summarizer.evaluation import (
    ROUGE_LABELS,
    ROUGE_TYPES,
    evaluate_dataset,
    load_reference_dataset,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the summarizer with ROUGE.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to reference JSON")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face BART checkpoint")
    p.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_SUMMARY_TOKENS)
    p.add_argument("--min-length", type=int, default=DEFAULT_MIN_SUMMARY_TOKENS)
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples")
    p.add_argument("--output", default=str(DEFAULT_RESULTS_PATH), help="Where to save JSON results")
    return p.parse_args(argv)


def print_table(title, scores):
    print(f"\n{title}")
    print(f"{'Metric':<10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    for t in ROUGE_TYPES:
        s = scores[t]
        print(f"{ROUGE_LABELS[t]:<10}{s['precision']:>12.4f}{s['recall']:>10.4f}{s['fmeasure']:>10.4f}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        samples = load_reference_dataset(args.dataset)
        if args.limit:
            samples = samples[: args.limit]

        from lecture_summarizer.summarizer import LectureSummarizer

        print(f"Loading model '{args.model}' (first run downloads it)...")
        summarizer = LectureSummarizer(args.model)

        def summarize(text):
            return summarizer.summarize(
                text,
                chunk_words=args.chunk_words,
                max_summary_tokens=args.max_length,
                min_summary_tokens=args.min_length,
            ).summary

        report = evaluate_dataset(
            summarize,
            samples,
            progress_callback=lambda d, t: print(f"  evaluated {d}/{t}", flush=True),
        )
    except LectureSummarizerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for item in report.per_sample:
        print_table(f"[{item['title']}]", item["scores"])
    print_table(f"AVERAGE over {len(report.per_sample)} samples", report.average)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "num_samples": len(report.per_sample),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **report.to_dict(),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
