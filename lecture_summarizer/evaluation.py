"""ROUGE-1 / ROUGE-2 / ROUGE-L evaluation helpers."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .config import DEFAULT_DATASET_PATH
from .errors import DatasetError

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
ROUGE_LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}

Scores = Dict[str, Dict[str, float]]


def _zero_scores() -> Scores:
    return {t: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for t in ROUGE_TYPES}


def compute_rouge(prediction: str, reference: str) -> Scores:
    """ROUGE-1/2/L precision, recall and F1 of ``prediction`` against ``reference``."""
    if not (prediction or "").strip() or not (reference or "").strip():
        return _zero_scores()
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "rouge-score is not installed. Run: pip install -r requirements.txt"
        ) from exc

    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    raw = scorer.score(reference, prediction)  # score(target, prediction)
    return {
        name: {"precision": s.precision, "recall": s.recall, "fmeasure": s.fmeasure}
        for name, s in raw.items()
    }


def average_scores(all_scores: List[Scores]) -> Scores:
    """Macro-average a list of score dicts."""
    if not all_scores:
        return _zero_scores()
    n = len(all_scores)
    return {
        t: {
            k: sum(s[t][k] for s in all_scores) / n
            for k in ("precision", "recall", "fmeasure")
        }
        for t in ROUGE_TYPES
    }


def scores_to_rows(scores: Scores, digits: int = 4) -> List[dict]:
    """Flatten scores into table rows for display."""
    return [
        {
            "Metric": ROUGE_LABELS[t],
            "Precision": round(scores[t]["precision"], digits),
            "Recall": round(scores[t]["recall"], digits),
            "F1": round(scores[t]["fmeasure"], digits),
        }
        for t in ROUGE_TYPES
    ]


def load_reference_dataset(path: Union[str, Path] = DEFAULT_DATASET_PATH) -> List[dict]:
    """Load samples: a list of {id, title, text, reference_summary}."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Reference dataset not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DatasetError(f"Could not read dataset {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise DatasetError("The dataset must be a non-empty JSON list.")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not all(
            str(item.get(k, "")).strip() for k in ("text", "reference_summary")
        ):
            raise DatasetError(
                f"Dataset item #{i} needs non-empty 'text' and 'reference_summary'."
            )
        item.setdefault("id", f"sample-{i + 1}")
        item.setdefault("title", item["id"])
    return data


@dataclass
class EvaluationReport:
    per_sample: List[dict] = field(default_factory=list)
    average: Scores = field(default_factory=_zero_scores)

    def to_dict(self) -> dict:
        return {"per_sample": self.per_sample, "average": self.average}


def evaluate_dataset(
    summarize_fn: Callable[[str], str],
    samples: List[dict],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> EvaluationReport:
    """Run ``summarize_fn`` on each sample and score it against its reference."""
    per_sample = []
    for i, sample in enumerate(samples, start=1):
        prediction = summarize_fn(sample["text"])
        scores = compute_rouge(prediction, sample["reference_summary"])
        per_sample.append(
            {
                "id": sample["id"],
                "title": sample["title"],
                "prediction": prediction,
                "reference": sample["reference_summary"],
                "scores": scores,
            }
        )
        if progress_callback:
            progress_callback(i, len(samples))
    return EvaluationReport(
        per_sample=per_sample,
        average=average_scores([p["scores"] for p in per_sample]),
    )
