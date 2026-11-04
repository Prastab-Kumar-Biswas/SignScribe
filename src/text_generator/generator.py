"""
SignScribe — Lightweight Sign-to-Text Generator.

Replaces complex rule mappers and heavy LLM inference with an ultra-fast,
deterministic, and clean text generator.

Features:
- Finger-spelling assembly: groups sequential letter/digit signs (A-Z, 0-9) into words.
- ISL word translation: cleans up vocabulary tokens (e.g., "GOOD_MORNING" -> "Good morning").
- Smart punctuation and sentence casing.
- Interactive controls: add space, backspace, and clear text.
"""

from __future__ import annotations

import re


# Word mappings for cleaner natural English output
WORD_DISPLAY_MAP: dict[str, str] = {
    "GOOD_MORNING": "Good morning",
    "GOOD_AFTERNOON": "Good afternoon",
    "THANK_YOU": "Thank you",
    "WHAT_IS_YOUR_NAME": "What is your name?",
    "HELLO": "Hello",
    "FEDUP": "fed up",
    "MATHS": "maths",
    "VEGETABLES": "vegetables",
}


class SignTextGenerator:
    """Converts a sequence of detected sign language tokens into formatted English text.

    Distinguishes between fingerspelled letters (which accumulate into single words)
    and whole-word signs (which append as separate words).
    """

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self._current_spelling: list[str] = []
        self._committed_words: list[str] = []

    def reset(self) -> None:
        """Clear all text and state."""
        self.tokens.clear()
        self._current_spelling.clear()
        self._committed_words.clear()

    def add_token(self, token: str) -> str:
        """Process a newly recognized sign token and return the updated text.

        Parameters
        ----------
        token : str
            Recognized gesture label (e.g. 'A', 'HELLO', 'WATER', '3').

        Returns
        -------
        str
            The current formatted text.
        """
        token = str(token).strip().upper()
        if not token or token == "IDLE":
            return self.get_text()

        self.tokens.append(token)

        # Check if the token is a single alphanumeric character (fingerspelling)
        if len(token) == 1 and token.isalnum():
            self._current_spelling.append(token)
        else:
            # Whole-word sign encountered: flush any in-progress spelled word first
            self._flush_spelling()
            display_word = WORD_DISPLAY_MAP.get(token, token.replace("_", " ").lower())
            self._committed_words.append(display_word)

        return self.get_text()

    def add_space(self) -> str:
        """Explicitly finish the current spelled word and insert a word break."""
        self._flush_spelling()
        return self.get_text()

    def backspace(self) -> str:
        """Delete the last character or word."""
        if self._current_spelling:
            self._current_spelling.pop()
        elif self._committed_words:
            self._committed_words.pop()

        if self.tokens:
            self.tokens.pop()

        return self.get_text()

    def _flush_spelling(self) -> None:
        """Commit currently accumulated letters as a word."""
        if self._current_spelling:
            word = "".join(self._current_spelling)
            self._committed_words.append(word)
            self._current_spelling.clear()

    def get_text(self) -> str:
        """Format committed words and current spelling into natural English text."""
        words: list[str] = list(self._committed_words)
        if self._current_spelling:
            words.append("".join(self._current_spelling))

        if not words:
            return ""

        # Join words with proper spacing
        result_parts: list[str] = []
        for i, w in enumerate(words):
            if not w:
                continue

            # Check if this word starts a sentence
            is_start_of_sentence = (
                i == 0
                or (result_parts and result_parts[-1].endswith((".", "?", "!")))
            )

            formatted = w.capitalize() if is_start_of_sentence else w
            result_parts.append(formatted)

        text = " ".join(result_parts)

        # Clean up punctuation spacing: e.g. " ?" -> "?"
        text = re.sub(r'\s+([?.!,])', r'\1', text)

        return text

    def get_raw_tokens(self) -> list[str]:
        """Return the sequence of all raw tokens recognized so far."""
        return list(self.tokens)

