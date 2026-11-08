from lecture_summarizer.chunking import chunk_text, split_sentences


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunks_respect_limit_and_keep_all_words():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, max_words=50)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_very_long_sentence_is_split():
    text = " ".join(["word"] * 130)
    chunks = chunk_text(text, max_words=50)
    assert [len(c.split()) for c in chunks] == [50, 50, 30]


def test_sentence_splitter():
    assert split_sentences("Hello world. This is a test! Is it? Yes.") == [
        "Hello world.",
        "This is a test!",
        "Is it?",
        "Yes.",
    ]
