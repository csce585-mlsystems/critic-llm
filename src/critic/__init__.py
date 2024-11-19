def hello() -> str:
    return "Hello from critic!"


import os

os.environ["KERAS_BACKEND"] = "jax"


from functools import cache
from critic.kbd_model import KbdModel
from critic.kbd_corrector import KbdCorrector
from transformers import BitsAndBytesConfig

from critic.lm_adapter import LMAdapter


def load_combo_corrector():
    model = KbdModel()

    model.load_weights("models/kbd_model.weights.h5")
    model.kbd_layout_distr = cache(model.kbd_layout_distr)
    kbd = KbdCorrector(model)
    corrector = LMAdapter("distilbert/distilgpt2", base=kbd)
    return corrector


def load_kbd_corrector():
    model = KbdModel()

    model.load_weights("models/kbd_model.weights.h5")
    model.kbd_layout_distr = cache(model.kbd_layout_distr)
    kbd = KbdCorrector(model)
    return kbd
