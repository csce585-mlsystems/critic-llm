from functools import cache
from spylls.hunspell import Dictionary

import numpy as np

from critic.corrector import Corrections, Corrector


class UniformCorrector(Corrector):
    """Considers every suggestion equally: useful as a base for other models."""

    def __init__(self, not_in_lexicon_prob: float = 0.01):
        self.dictionary = Dictionary.from_files("en_US")
        self.suggester = self.dictionary.suggester
        self.not_in_lexicon_prob = not_in_lexicon_prob

    def push_word(self, word: str):
        return

    def clear_context(self):
        return

    @cache
    def correct(self, word: str) -> Corrections:
        words = [word] + [
            sug.text for sug in self.suggester.suggestions(word) if sug.text != word
        ]

        probs = np.ones(len(words))

        if not self.dictionary.lookup(word):
            probs[0] *= self.not_in_lexicon_prob

        probs /= sum(probs)
        return Corrections(words, probs)


class SimpleCorrector(UniformCorrector):
    """Uses a simple spell checker that only looks at edit distance."""

    def __init__(self, gamma: float = 0.8, not_in_lexicon_prob: float = 0.01):
        super().__init__(not_in_lexicon_prob)
        self.gamma = gamma

    def push_word(self, word: str):
        return

    def clear_context(self):
        return

    def correct(self, word: str) -> Corrections:
        probs = super().correct(word).as_series()

        scale = self.gamma ** np.arange(len(probs))

        probs *= scale
        probs /= sum(probs)
        return Corrections(probs.index, probs.values)


if __name__ == "__main__":
    checker = SimpleCorrector()

    corrections = checker.correct("tere")
    print(corrections.as_series())
