"""Custom exceptions so the UI/CLI can show friendly messages."""


class LectureSummarizerError(Exception):
    """Base class for all expected, user-facing errors."""


class InvalidPDFError(LectureSummarizerError):
    """The uploaded file is not a readable PDF (corrupt, empty, wrong type)."""


class EncryptedPDFError(LectureSummarizerError):
    """The PDF is password protected."""


class EmptyTextError(LectureSummarizerError):
    """No text could be extracted (e.g. a scanned/image-only PDF)."""


class TextTooShortError(LectureSummarizerError):
    """The text is too short to produce a meaningful summary."""


class DatasetError(LectureSummarizerError):
    """The reference dataset is missing or malformed."""
