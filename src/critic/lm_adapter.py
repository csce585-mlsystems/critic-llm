"""Interface for a language model that can give logits."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from critic.corrector import Corrections, Corrector
from critic.simple_corrector import UniformCorrector
import jax
import jax.numpy as jnp


class LMAdapter(Corrector):
    """Modifies an existing corrector to incorporate signal from a language model."""

    def __init__(
        self,
        model_name,
        base: Corrector | None = None,
        kbd_weight: float = 1,
        leading_space: bool = True,
        model: AutoModelForCausalLM | None = None,
    ):
        self.model_name = model_name
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.leading_space = leading_space
        self.prefix = []
        if base is None:
            self.base = UniformCorrector()
        else:
            self.base = base

        self.kbd_weight = kbd_weight

    def push_word(self, word: str):
        self.prefix.append(word)
        self.base.push_word(word)

    def clear_context(self):
        self.prefix = []
        self.base.clear_context()

    def correct(self, word: str) -> Corrections:
        prefix = self.tokenizer.bos_token + " ".join(self.prefix)
        if not self.leading_space:
            prefix += " "

        prefix_tokens = self.tokenizer([prefix], return_tensors="pt")

        # seed cache?
        # self.model(**prefix_tokens)

        corrs = self.base.correct(word)

        if self.leading_space:
            corr_words = [" " + word for word in corrs.words]
        else:
            corr_words = list(corrs.words)

        tokens = self.tokenizer(
            [prefix + word for word in corr_words],
            return_tensors="pt",
            padding="longest",
        )

        num_prefix = prefix_tokens["input_ids"].shape[-1]

        out = self.model(**tokens, return_dict=True)

        # print(out.logits.shape)

        logits = out.logits[..., num_prefix - 1 : -1, :].numpy(force=True)
        pad_mask = tokens.attention_mask[..., num_prefix:].numpy(force=True)

        labels = tokens.input_ids[..., num_prefix:].numpy(force=True)

        # print(labels.shape, logits.shape)

        log_probs = jax.vmap(jax.vmap(lambda a, b: a[b]))(
            jax.nn.log_softmax(logits, axis=-1), labels
        )

        losses = jnp.sum(log_probs * pad_mask, axis=-1)

        probs = jax.nn.softmax(
            jnp.log(jnp.array(corrs.probs)) * self.kbd_weight + losses
        )

        return Corrections(corrs.words, probs)


if __name__ == "__main__":
    from critic.kbd_model import KbdModel
    from critic.kbd_corrector import KbdCorrector
    from transformers import BitsAndBytesConfig

    model = KbdModel()

    model.load_weights("models/kbd_model.weights.h5")
    kbd = KbdCorrector(model)
    corrector = LMAdapter("distilbert/distilgpt2", base=kbd)

    # corrector.push_words("The coverage about me in the paper gas")

    def show(context, word):
        corrector.clear_context()
        corrector.push_words(context)
        corrections = corrector.correct(word)
        print(corrections.as_series())

        print(corrector.base.correct(word).as_series())

        print(corrector.base.base.correct(word).as_series())

    show("", "Thr")
    show("The coverage about me in the paper", "gas")
