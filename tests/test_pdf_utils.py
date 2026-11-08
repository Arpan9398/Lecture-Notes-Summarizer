import pytest

from lecture_summarizer.errors import EmptyTextError, InvalidPDFError
from lecture_summarizer.pdf_utils import clean_text, extract_text_from_pdf


def test_clean_text_joins_hyphenation_and_lines():
    raw = "This is an exam-\nple of text\nwrapped  over lines.\n\nNew paragraph."
    assert clean_text(raw) == "This is an example of text wrapped over lines.\n\nNew paragraph."


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(" \n\n ") == ""


def test_invalid_bytes_raise_invalid_pdf():
    pytest.importorskip("fitz")
    with pytest.raises(InvalidPDFError):
        extract_text_from_pdf(b"this is definitely not a pdf")


def test_empty_bytes_raise_invalid_pdf():
    pytest.importorskip("fitz")
    with pytest.raises(InvalidPDFError):
        extract_text_from_pdf(b"")


def test_valid_pdf_roundtrip():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Lecture one covers photosynthesis.")
    data = doc.tobytes()
    doc.close()
    assert "photosynthesis" in extract_text_from_pdf(data)


def test_blank_pdf_raises_empty_text():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    with pytest.raises(EmptyTextError):
        extract_text_from_pdf(data)
