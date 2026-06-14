"""Self-contained readability scoring (Flesch-Kincaid grade level).

A deliberate dependency choice: rather than pull in textstat/NLTK (which downloads
a syllable corpus at runtime and fails in locked-down environments), this
implements Flesch-Kincaid directly with a vowel-group syllable estimator. The
estimate is approximate at the single-word level but stable and reproducible in
aggregate, which is what a target-reading-level metric needs.

Flesch-Kincaid grade = 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z]+")
_SENTENCE = re.compile(r"[.!?]+")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    w = word.lower()
    groups = _VOWEL_GROUP.findall(w)
    n = len(groups)
    # silent trailing 'e' rarely forms its own syllable
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _counts(text: str) -> tuple[int, int, int]:
    words = _WORD.findall(text)
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    n_words = len(words)
    n_sentences = max(1, len(sentences))
    n_syllables = sum(count_syllables(w) for w in words)
    return n_words, n_sentences, n_syllables


def flesch_kincaid_grade(text: str) -> float:
    n_words, n_sentences, n_syllables = _counts(text)
    if n_words == 0:
        return 0.0
    grade = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
    return round(grade, 2)


def flesch_reading_ease(text: str) -> float:
    n_words, n_sentences, n_syllables = _counts(text)
    if n_words == 0:
        return 0.0
    score = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
    return round(score, 2)
