# Lecture Notes Summarizer

Upload a PDF of lecture notes and get a concise, abstractive summary from a pretrained
**BART** model, with **ROUGE-1 / ROUGE-2 / ROUGE-L** evaluation and a **Streamlit** UI.

**Stack:** Python · Hugging Face Transformers (BART) · PyMuPDF · rouge-score · Streamlit

## How it works

1. **Extract** text from the PDF with PyMuPDF and clean it (hyphenation, line breaks, whitespace).
2. **Chunk** the text into sentence-aligned pieces of at most ~500 words (BART accepts 1024 tokens).
3. **Summarize** each chunk with `facebook/bart-large-cnn`, then condense the chunk summaries
   into one final summary (repeated if they are still too long).
4. **Evaluate** summaries against reference summaries with ROUGE.

## Project structure

```
lecture-notes-summarizer/
├── app.py                         # Streamlit UI
├── evaluate.py                    # CLI ROUGE evaluation
├── lecture_summarizer/
│   ├── config.py                  # paths, model name, limits
│   ├── errors.py                  # friendly custom exceptions
│   ├── pdf_utils.py               # PyMuPDF extraction + text cleaning
│   ├── chunking.py                # sentence-aware chunking
│   ├── summarizer.py              # BART chunked summarization
│   └── evaluation.py              # ROUGE + dataset helpers
├── data/
│   ├── reference/reference_summaries.json   # 5 texts + reference summaries
│   └── sample/sample_lecture_notes.pdf      # PDF to try in the UI
├── tests/                         # pytest tests
├── requirements.txt
└── requirements-dev.txt           # + pytest
```

## Setup

Requires Python 3.9+ (3.10–3.12 recommended).

```bash
cd lecture-notes-summarizer

python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> The first time the model is used, `facebook/bart-large-cnn` (~1.6 GB) is downloaded from the
> Hugging Face Hub into your user cache (`~/.cache/huggingface`), **not** into this project.
> An internet connection is needed for that first run. A GPU is optional; CPU works but is slower.

## Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown (usually http://localhost:8501), then:

1. Upload a PDF (try `data/sample/sample_lecture_notes.pdf`, also downloadable in the sidebar).
2. Review the **original text preview**.
3. Click **Summarize**.
4. Read the **generated summary**. Optionally paste a reference summary to see ROUGE scores.

The **Evaluate on sample dataset** tab runs the model on the bundled reference data and shows
average and per-sample ROUGE.

## Run the evaluation script

```bash
python evaluate.py                 # all samples
python evaluate.py --limit 2       # quick run
python evaluate.py --help          # more options (model, lengths, output path)
```

It prints ROUGE-1/2/L precision, recall and F1 per sample and on average, and saves a JSON report
to `results/evaluation_results.json`. Scores depend on your hardware/library versions and the
model, so run it yourself to get real numbers; none are hard-coded here.

### Reference dataset

`data/reference/reference_summaries.json` is a list of
`{"id", "title", "text", "reference_summary"}` objects. It is deliberately tiny (5 hand-written
examples) for demonstration. Add your own entries to evaluate on your lecture material.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Error handling

- Corrupt, empty or non-PDF uploads → clear error message.
- Password-protected PDFs → clear error message.
- PDFs without extractable text (scans) → message suggesting OCR.
- Text under 30 words → message instead of a meaningless summary.

## Notes and limitations

- `facebook/bart-large-cnn` was fine-tuned on news articles; lecture notes with heavy math,
  tables or slides may summarize less well. Any BART summarization checkpoint can be entered in
  the sidebar (or passed with `--model`).
- Scanned PDFs need OCR first; this project does not include OCR.
- ROUGE measures word overlap with a reference, not factual accuracy, and 5 samples is far too
  few for general conclusions.
