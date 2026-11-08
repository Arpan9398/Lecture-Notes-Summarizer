"""Streamlit UI for the Lecture Notes Summarizer.

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from lecture_summarizer.config import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_DATASET_PATH,
    DEFAULT_MAX_SUMMARY_TOKENS,
    DEFAULT_MIN_SUMMARY_TOKENS,
    DEFAULT_MODEL_NAME,
    PREVIEW_CHARS,
    SAMPLE_PDF_PATH,
)
from lecture_summarizer.errors import LectureSummarizerError
from lecture_summarizer.evaluation import (
    ROUGE_LABELS,
    ROUGE_TYPES,
    compute_rouge,
    evaluate_dataset,
    load_reference_dataset,
    scores_to_rows,
)
from lecture_summarizer.pdf_utils import extract_text_from_pdf

st.set_page_config(page_title="Lecture Notes Summarizer", page_icon="📝", layout="wide")


@st.cache_resource(show_spinner="Loading BART model (the first run downloads ~1.6 GB)...")
def get_summarizer(model_name: str):
    from lecture_summarizer.summarizer import LectureSummarizer

    return LectureSummarizer(model_name)


def show_rouge(scores: dict) -> None:
    """Metric cards (F1) plus a precision/recall/F1 table."""
    rows = scores_to_rows(scores)
    for col, row in zip(st.columns(len(rows)), rows):
        col.metric(f"{row['Metric']} F1", f"{row['F1']:.4f}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("BART model", value=DEFAULT_MODEL_NAME)
    chunk_words = st.slider("Chunk size (words)", 200, 800, DEFAULT_CHUNK_WORDS, step=50)
    max_tokens = st.slider("Max summary length (tokens)", 60, 300, DEFAULT_MAX_SUMMARY_TOKENS, step=10)
    min_tokens = st.slider("Min summary length (tokens)", 10, 100, DEFAULT_MIN_SUMMARY_TOKENS, step=5)
    if min_tokens >= max_tokens:
        st.warning("Min length should be smaller than max length.")
    if SAMPLE_PDF_PATH.is_file():
        st.download_button(
            "Download sample lecture PDF",
            data=SAMPLE_PDF_PATH.read_bytes(),
            file_name=SAMPLE_PDF_PATH.name,
            mime="application/pdf",
        )

st.title("📝 Lecture Notes Summarizer")
st.caption("Upload lecture notes as a PDF and get an abstractive summary from a pretrained BART model.")

tab_summarize, tab_eval = st.tabs(["Summarize PDF", "Evaluate on sample dataset"])

# ------------------------------------------------------------ summarize tab
with tab_summarize:
    uploaded = st.file_uploader("Upload a PDF of lecture notes", type=["pdf"])

    if uploaded is None:
        st.info("Upload a PDF to begin. You can download a sample PDF from the sidebar.")
    else:
        data = uploaded.getvalue()
        file_key = f"{uploaded.name}-{len(data)}"
        text = None
        try:
            text = extract_text_from_pdf(data)
        except LectureSummarizerError as exc:
            st.error(str(exc))
        except Exception as exc:  # unexpected
            st.error(f"Unexpected error while reading the PDF: {exc}")

        if text:
            n_words = len(text.split())
            st.subheader("Original text preview")
            st.caption(f"{n_words:,} words extracted.")
            st.text_area(
                "Original text",
                value=text[:PREVIEW_CHARS] + ("\n\n[... preview truncated ...]" if len(text) > PREVIEW_CHARS else ""),
                height=280,
                disabled=True,
                label_visibility="collapsed",
            )

            reference = st.text_area(
                "Reference summary (optional, enables ROUGE scores)",
                placeholder="Paste a human-written summary to compare against...",
                height=100,
            )

            if st.button("Summarize", type="primary"):
                try:
                    summarizer = get_summarizer(model_name)
                    progress = st.progress(0.0, text="Summarizing...")
                    result = summarizer.summarize(
                        text,
                        chunk_words=chunk_words,
                        max_summary_tokens=max_tokens,
                        min_summary_tokens=min_tokens,
                        progress_callback=lambda d, t: progress.progress(
                            d / t, text=f"Summarizing... step {d}/{t}"
                        ),
                    )
                    progress.empty()
                    st.session_state["result"] = {"file_key": file_key, "result": result}
                except LectureSummarizerError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Summarization failed: {exc}")

            stored = st.session_state.get("result")
            if stored and stored["file_key"] == file_key:
                result = stored["result"]
                st.subheader("Generated summary")
                st.success(result.summary)
                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks", result.num_chunks)
                c2.metric("Input words", f"{result.input_words:,}")
                c3.metric("Summary words", result.summary_words)
                st.download_button("Download summary (.txt)", result.summary, file_name="summary.txt")

                if reference.strip():
                    st.subheader("ROUGE scores vs. your reference")
                    try:
                        show_rouge(compute_rouge(result.summary, reference))
                    except Exception as exc:
                        st.error(f"Could not compute ROUGE: {exc}")
                else:
                    st.caption("Paste a reference summary above to see ROUGE-1, ROUGE-2 and ROUGE-L.")

# ---------------------------------------------------------- evaluation tab
with tab_eval:
    st.write(
        "Runs the summarizer on the small bundled reference dataset "
        f"(`{DEFAULT_DATASET_PATH.relative_to(DEFAULT_DATASET_PATH.parents[2])}`) "
        "and reports ROUGE-1, ROUGE-2 and ROUGE-L against the reference summaries."
    )
    if st.button("Run evaluation"):
        try:
            samples = load_reference_dataset()
            summarizer = get_summarizer(model_name)
            progress = st.progress(0.0, text="Evaluating...")
            report = evaluate_dataset(
                lambda t: summarizer.summarize(
                    t,
                    chunk_words=chunk_words,
                    max_summary_tokens=max_tokens,
                    min_summary_tokens=min_tokens,
                ).summary,
                samples,
                progress_callback=lambda d, t: progress.progress(d / t, text=f"Evaluated {d}/{t}"),
            )
            progress.empty()
            st.session_state["eval_report"] = report
        except LectureSummarizerError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    report = st.session_state.get("eval_report")
    if report:
        st.subheader(f"Average ROUGE over {len(report.per_sample)} samples")
        show_rouge(report.average)
        st.subheader("Per-sample F1")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Sample": p["title"], **{ROUGE_LABELS[t]: round(p["scores"][t]["fmeasure"], 4) for t in ROUGE_TYPES}}
                    for p in report.per_sample
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Generated vs. reference summaries"):
            for p in report.per_sample:
                st.markdown(f"**{p['title']}**")
                st.markdown(f"*Generated:* {p['prediction']}")
                st.markdown(f"*Reference:* {p['reference']}")
