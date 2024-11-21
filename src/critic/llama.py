"""
Llama-cpp model.
"""

from os import PathLike
from pathlib import Path
from llama_cpp import Llama
from scipy.special import log_softmax, softmax


from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from critic.corrector import Corrections, Corrector
from critic.simple_corrector import UniformCorrector


class LlamaAdapter(Corrector):
    """Modifies an existing corrector to incorporate signal from a language model."""

    def __init__(
        self,
        model: Llama | PathLike,
        base: Corrector | None = None,
        kbd_weight: float = 1,
        leading_space: bool = True,
        cache_state: bool = False,
        **llama_kwargs,
    ):
        kwargs = dict(
            logits_all=True,
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            n_ctx=64,  # Uncomment to increase the context window
            verbose=False,
        )
        kwargs.update(llama_kwargs)
        if not isinstance(model, Llama):
            self.model = Llama(model_path=model, **kwargs)
        else:
            self.model = model

        self.leading_space = leading_space
        self.prefix = []
        if base is None:
            self.base = UniformCorrector()
        else:
            self.base = base

        self.kbd_weight = kbd_weight
        self.cache_state = cache_state

    def push_word(self, word: str):
        self.prefix.append(word)
        self.base.push_word(word)

    def clear_context(self):
        self.prefix = []
        self.base.clear_context()

    def correct(self, word: str) -> Corrections:
        prefix = " ".join(self.prefix)
        if not self.leading_space:
            prefix += " "

        prefix_tokens = self.model.tokenize(prefix.encode("utf-8"))

        corrs = self.base.correct(word)

        if self.leading_space:
            corr_words = [" " + word for word in corrs.words]
        else:
            corr_words = list(corrs.words)

        suffixes = [
            self.model.tokenize(word.encode("utf-8"), add_bos=False)
            for word in corr_words
        ]

        num_prefix = len(prefix_tokens)

        if self.cache_state:
            self.model.reset()
            self.model.eval(prefix_tokens)
            state = self.model.save_state()

        opt_scores = []
        for suffix in suffixes:
            if self.cache_state:
                self.model.load_state(state)
                if len(suffix) > 1:
                    self.model.eval(suffix[:-1])
            else:
                self.model.reset()
                self.model.eval(prefix_tokens + suffix)

            logits = self.model.scores[num_prefix - 1 : num_prefix + len(suffix) - 1, :]
            labels = suffix

            log_probs = log_softmax(logits, axis=1)
            scores = log_probs[np.arange(len(labels)), labels]

            score = np.sum(scores)
            opt_scores.append(score)

        probs = softmax(np.log(corrs.probs) * self.kbd_weight + np.array(opt_scores))

        return Corrections(corrs.words, probs)


if __name__ == "__main__":
    from critic.kbd_model import KbdModel
    from critic.kbd_corrector import KbdCorrector
    from time import perf_counter

    model = KbdModel()

    model.load_weights("models/kbd_model.weights.h5")
    kbd = KbdCorrector(model)

    llm = Llama(
        "models/llama-3.2-1b-q4_k_m.gguf",
        n_ctx=64,
        n_threads=2,
        n_threads_batch=2,
        logits_all=True,
        flash_attn=True,
    )
    corrector = LlamaAdapter(llm, base=kbd, cache_state=True)

    # corrector.push_words("The coverage about me in the paper gas")

    def show(context, word):
        corrector.clear_context()
        corrector.push_words(context)

        start = perf_counter()
        corrections = corrector.correct(word)
        end = perf_counter()

        print(f"Time taken: {(end - start):.2f}")
        print(corrections.as_series())

        print(corrector.base.correct(word).as_series())

        # print(corrector.base.base.correct(word).as_series())

    show("", "Thr")
    show("The coverage about me in the", "paper")
    show("The coverage about", "he")
    show("The coverage about me in the paper", "gas")

    corrector.model.close()
