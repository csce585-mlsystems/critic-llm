from dataclasses import dataclass
from typing import Sequence
import pandas as pd


@dataclass
class Corrections:
    """Possible corrections for a single word."""

    # Words to correct to.
    words: Sequence[str]

    # Probabilities for each word.
    probs: Sequence[float]

    def as_series(self) -> pd.Series:
        """Returns a sorted pd.Series."""
        return pd.Series(self.probs, self.words).sort_values(ascending=False)

    def truncate(self, i: int) -> "Corrections":
        """Truncates to the first i corrections."""
        return Corrections(self.words[:i], self.probs[:i])


class Corrector:
    """A model for correcting words in context."""

    def push_word(self, word: str):
        """Adds the (correct) word to the existing context."""
        raise NotImplementedError()

    def push_words(self, words: str):
        for word in words.split(" "):
            self.push_word(word)

    def clear_context(self):
        """Resets context."""
        raise NotImplementedError()

    def correct(self, word: str) -> Corrections:
        """Computes potential corrections for a word, given previous context."""
        raise NotImplementedError()
