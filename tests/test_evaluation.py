import pytest

from lecture_summarizer.errors import DatasetError
from lecture_summarizer.evaluation import (
    average_scores,
    compute_rouge,
    evaluate_dataset,
    load_reference_dataset,
)


def test_reference_dataset_is_valid():
    samples = load_reference_dataset()
    assert len(samples) >= 3
    for s in samples:
        assert s["text"].strip() and s["reference_summary"].strip()
        assert len(s["reference_summary"].split()) < len(s["text"].split())


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(DatasetError):
        load_reference_dataset(tmp_path / "nope.json")


def test_empty_prediction_scores_zero():
    scores = compute_rouge("", "some reference")
    assert scores["rouge1"]["fmeasure"] == 0.0


def test_identical_text_scores_one():
    pytest.importorskip("rouge_score")
    scores = compute_rouge("the cat sat on the mat", "the cat sat on the mat")
    assert all(abs(scores[t]["fmeasure"] - 1.0) < 1e-9 for t in scores)


def test_evaluate_dataset_with_dummy_summarizer():
    pytest.importorskip("rouge_score")
    samples = load_reference_dataset()[:2]
    # "Summarizer" that returns the reference itself -> perfect scores.
    lookup = {s["text"]: s["reference_summary"] for s in samples}
    report = evaluate_dataset(lambda t: lookup[t], samples)
    assert abs(report.average["rouge1"]["fmeasure"] - 1.0) < 1e-9
    assert len(report.per_sample) == 2


def test_average_scores_empty():
    assert average_scores([])["rougeL"]["fmeasure"] == 0.0
