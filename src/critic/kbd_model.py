"""
Keyboard model: computes likelihood of different typing errors.
"""

from dataclasses import dataclass
import os
from typing import Any, Sequence

from critic.kbd_layout import QWERTY

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers as nn
from keras import ops
import jax
import jax.numpy as jnp
from scipy.stats import multivariate_normal
# import tensorflow_probability.substrates.jax as tfp

# tfd = tfp.distributions

# TODO this needs to be included in information given if we want to support different layouts
VOWELS = ops.array([list(QWERTY.get_xy(c)) for c in "aeiou"])


@dataclass
class MultivariateNormal:
    loc: Sequence[Any]
    cov: Sequence[Any]

    def log_prob(self, x0):
        def f(x):
            loc, cov = self.loc, self.cov
            y = (x - loc).T @ ops.inv(cov) @ (x - loc)
            Z = (2 * jnp.pi) ** (0.5 * cov.shape[0]) * ops.abs(ops.det(cov)) ** 0.5
            return ops.log(ops.exp(-0.5 * y) / Z)

        return jax.vmap(f)(x0)


class KbdModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.tilt = self.add_weight((), initializer="zeros", name="tilt")
        self.aspect_ratio = self.add_weight(
            (), initializer="zeros", name="aspect_ratio"
        )
        # omission, insertion, substitute, transpose
        self.category_logits = self.add_weight((4,), initializer="normal", name="oist")
        self.two_hand_logit = self.add_weight((), initializer="zeros", name="two_hand")
        self.vowel_logit = self.add_weight((), initializer="zeros", name="vowel")
        self.scale = self.add_weight((), initializer="zeros", name="scale")
        self.power = self.add_weight(
            (),
            initializer=lambda shape, dtype: 2
            * keras.initializers.Ones()(shape, dtype),
            name="power",
        )

    def kbd_layout_distr(self):
        x = ops.sigmoid(self.aspect_ratio)
        y = 1 - x

        cov = ops.diag([x, y]) * ops.exp(self.scale)

        cost = ops.cos(self.tilt)
        sint = ops.sin(self.tilt)

        R = ops.convert_to_tensor([[cost, -sint], [sint, cost]])
        # R = torch.stack([cost, -sint, sint, cost]).reshape(2, 2)

        # dist = tfd.MultivariateNormalFullCovariance(
        #     loc=ops.zeros([2], dtype="float32"),
        #     covariance_matrix=ops.matmul(ops.matmul(R, cov), R.T),
        # )
        dist = MultivariateNormal(
            loc=ops.zeros([2], dtype="float32"),
            cov=ops.matmul(ops.matmul(R, cov), R.T),
        )
        return dist

    def kbd_layout_dist(self, delta):
        return self.kbd_layout_distr().log_prob(
            ops.sqrt(ops.abs(delta) + 1e-6) ** (self.power)
        )

    def category_probs(self):
        return ops.log_softmax(self.category_logits)

    def insertion_log_prob(self, wrong_xy, right_xy):
        delta = right_xy - wrong_xy
        return self.kbd_layout_dist(delta)

    def omission_log_prob(self, wrong_xy, right_xy):
        return self.kbd_layout_dist(wrong_xy * 0)

    def substitute_log_prob(self, wrong_xy, right_xy):
        delta = right_xy - wrong_xy
        wrong_vowel = (
            (wrong_xy[..., None, :] == VOWELS[None, ...]).all(axis=-1).any(axis=-1)
        )
        right_vowel = (
            (right_xy[..., None, :] == VOWELS[None, ...]).all(axis=-1).any(axis=-1)
        )
        vowel_prob = wrong_vowel & right_vowel

        a, b = ops.log_sigmoid(self.vowel_logit), ops.log_sigmoid(-self.vowel_logit)
        vowel_prob = ops.where(vowel_prob, a, b)

        return vowel_prob + self.kbd_layout_dist(delta)

    def transpose_log_prob(self, wrong_xy, right_xy):
        delta = right_xy - wrong_xy
        wrong_lh = wrong_xy[..., 0] <= 4
        right_lh = right_xy[..., 0] <= 4
        two_hand_prob = wrong_lh != right_lh

        a, b = (
            ops.log_sigmoid(self.two_hand_logit),
            ops.log_sigmoid(-self.two_hand_logit),
        )
        two_hand_prob = ops.where(two_hand_prob, a, b)

        return two_hand_prob + self.kbd_layout_dist(
            delta * 0
        )  # self.gaussian().log_prob(delta)

    def log_prob(self, kind, wrong_xy, right_xy):
        cond_probs = []
        for meth in (
            self.omission_log_prob,
            self.insertion_log_prob,
            self.substitute_log_prob,
            self.transpose_log_prob,
        ):
            cond_probs.append(meth(wrong_xy, right_xy))
            # print(cond_probs[-1].shape)

        probs = ops.array(cond_probs) + self.category_probs()[..., None]
        # print(cond_probs.shape)

        probs = probs * (ops.arange(4)[..., None] == kind[None, ...])
        probs = probs.sum(axis=0)

        return probs

    def call(self, inputs):
        kind, wrong_xy, right_xy = inputs
        return -self.log_prob(kind, wrong_xy, right_xy)


if __name__ == "__main__":
    import pandas as pd

    layout = QWERTY

    df = pd.read_feather("precomputed/all_typos.feather")
    metadata = pd.read_csv(
        "data/keystrokes/metadata_participants.txt", sep="\t"
    ).set_index("PARTICIPANT_ID")
    df = df[
        df["wrong_char"].str.lower().isin(set(layout.letters))
        & df["right_char"].str.lower().isin(set(layout.letters))
    ]
    df = df.reset_index(drop=True)
    df["wrong_xy"] = df["wrong_char"].apply(layout.get_xy)
    df["right_xy"] = df["right_char"].apply(layout.get_xy)
    df[["wrong_x", "wrong_y"]] = pd.DataFrame(df["wrong_xy"].tolist())
    df[["right_x", "right_y"]] = pd.DataFrame(df["right_xy"].tolist())
    df["kind_code"] = pd.Categorical(
        df["kind"], categories=("omission", "insertion", "substitute", "transpose")
    ).codes
    counts = df["right_char"].str.lower().value_counts()
    df["freq"] = (sum(counts) / counts[df["right_char"].str.lower()]).values

    mod = KbdModel()

    def process_inputs(subs):
        kind = ops.array(subs["kind_code"])
        wrong_xy = ops.array(list(map(list, subs["wrong_xy"])))
        right_xy = ops.array(list(map(list, subs["right_xy"])))
        return kind, wrong_xy, right_xy

    # mod.insertion_log_prob(*process_inputs(df.iloc[:64])[1:])
    print(mod.log_prob(*process_inputs(df.iloc[:64])))

    from keras.optimizers.schedules import PolynomialDecay

    subs = df.iloc[::]
    inputs = process_inputs(subs)
    weights = ops.array(subs["weight"]) * ops.array(subs["freq"])

    weights = weights / ops.mean(weights)

    def fit(epochs=25, valid_split=0.05, batch_size=256):
        mod = KbdModel()

        steps_in_epoch = round(
            (inputs[0].shape[0] * (1 - valid_split)) / (batch_size) + 0.5
        )

        decay_steps = steps_in_epoch * epochs

        def log_prob_loss(y_true, y_pred):
            return y_pred

        mod.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=PolynomialDecay(
                    4e-3, decay_steps, end_learning_rate=1e-6
                ),
                global_clipnorm=3.0,
            ),
            loss=log_prob_loss,
        )

        history = mod.fit(
            inputs,
            inputs[0] * 0,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=valid_split,
            sample_weight=weights,
        )

        print(pd.DataFrame(history.history))

        return mod

    mod = fit()
    print(mod.get_state_tree())
    mod.save_weights("models/kbd_model.weights.h5")
