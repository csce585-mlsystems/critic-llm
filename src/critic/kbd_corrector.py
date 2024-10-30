"""Keyboard-aware corrector."""

from critic.corrector import Corrections, Corrector
from critic.kbd_layout import QWERTY, KbdLayout
from critic.kbd_model import KbdModel
from critic.simple_corrector import UniformCorrector

import numpy as np
from scipy.special import softmax

from critic.string_alignment import kbd_log_prob


class KbdCorrector(Corrector):
    """Considers every suggestion equally: useful as a base for other models."""

    def __init__(
        self, model: KbdModel, base: Corrector | None = None, layout: KbdLayout = QWERTY
    ):
        self.model = model
        if base is None:
            self.base = UniformCorrector()
        else:
            self.base = base
        self.layout = layout

    def push_word(self, word: str):
        self.base.push_word(word)

    def clear_context(self):
        self.base.clear_context()

    def correct(self, word: str) -> Corrections:
        corrs = self.base.correct(word)

        kbd_probs = softmax(
            np.array([kbd_log_prob(corr, word, self.model) for corr in corrs.words])
        )

        probs = np.array(corrs.probs) * kbd_probs
        probs /= sum(probs)
        return Corrections(corrs.words, probs)


if __name__ == "__main__":
    model = KbdModel()

    model.load_weights("models/kbd_model.weights.h5")
    corrector = KbdCorrector(model)

    corrections = corrector.correct("tere")
    print(corrections.as_series())

    print(corrector.base.correct("tere").as_series())
