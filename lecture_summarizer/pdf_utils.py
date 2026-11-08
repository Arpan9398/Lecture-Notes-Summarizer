"""PDF text extraction with PyMuPDF."""

import re
from pathlib import Path
from typing import Union

from .errors import EmptyTextError, EncryptedPDFError, InvalidPDFError


def clean_text(text: str) -> str:
    """Normalize raw PDF text.

    - removes NUL characters
    - re-joins words hyphenated across line breaks ("exam-\\nple" -> "example")
    - turns single line breaks into spaces but keeps blank-line paragraph breaks
    - collapses repeated whitespace
    """
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(source: Union[bytes, bytearray, str, Path]) -> str:
    """Extract and clean text from a PDF given as bytes or a file path.

    Raises:
        InvalidPDFError: file is empty, corrupt or not a PDF.
        EncryptedPDFError: PDF needs a password.
        EmptyTextError: PDF has no extractable text (e.g. scanned images).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            raise InvalidPDFError("The uploaded file is empty.")
        open_kwargs = {"stream": bytes(source), "filetype": "pdf"}
    else:
        path = Path(source)
        if not path.is_file():
            raise InvalidPDFError(f"File not found: {path}")
        open_kwargs = {"filename": str(path)}

    try:
        doc = fitz.open(**open_kwargs)
    except Exception as exc:
        raise InvalidPDFError(
            "Could not open the file as a PDF. It may be corrupt or not a PDF."
        ) from exc

    try:
        if doc.needs_pass or doc.is_encrypted:
            raise EncryptedPDFError(
                "This PDF is password protected. Please upload an unlocked copy."
            )
        if doc.page_count == 0:
            raise InvalidPDFError("The PDF has no pages.")
        try:
            pages = [page.get_text("text") for page in doc]
        except Exception as exc:
            raise InvalidPDFError("The PDF is damaged and could not be read.") from exc
    finally:
        doc.close()

    text = clean_text("\n\n".join(pages))
    if not text:
        raise EmptyTextError(
            "No text found in this PDF. It may be a scanned document (images only); "
            "run OCR on it first and try again."
        )
    return text
