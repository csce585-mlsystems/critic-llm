def hello() -> str:
    return "Hello from critic!"


import os

os.environ["KERAS_BACKEND"] = "jax"


from functools import cache

from critic.kbd_model import KbdModel
from critic.kbd_corrector import KbdCorrector


def load_combo_corrector():
    from critic.lm_adapter import LMAdapter

    model = KbdModel()

    model.load_weights("models/kbd_model.weights.h5")
    model.kbd_layout_distr = cache(model.kbd_layout_distr)
    kbd = KbdCorrector(model)
    corrector = LMAdapter("distilbert/distilgpt2", base=kbd)
    return corrector


def load_llama_corrector(n_threads=2, new_kbd=True, kbd_weight=1, **kwargs):
    from critic.llama import Llama, LlamaAdapter

    model = KbdModel()

    path = "kbd_model_new" if new_kbd else "kbd_model"
    model.load_weights(f"models/{path}.weights.h5")
    model.kbd_layout_distr = cache(model.kbd_layout_distr)
    llm = Llama(
        "models/llama-3.2-1b-q4_k_m.gguf",
        n_ctx=64,
        n_threads=n_threads,
        n_threads_batch=n_threads,
        logits_all=True,
        flash_attn=True,
        verbose=False,
        **kwargs,
    )
    kbd = KbdCorrector(model)
    corrector = LlamaAdapter(llm, base=kbd, kbd_weight=kbd_weight, cache_state=True)
    return corrector


def load_kbd_corrector():
    model = KbdModel()

    model.load_weights("models/kbd_model.weights.h5")
    model.kbd_layout_distr = cache(model.kbd_layout_distr)
    kbd = KbdCorrector(model)
    return kbd
