from collections import defaultdict
from dataclasses import dataclass
from typing import Literal
import numpy as np
import jax
import torch
from critic.kbd_layout import QWERTY
from critic.kbd_model import KbdModel
from keras import ops


EditKind = Literal["omit", "insert", "sub", "transpose"]

EOS = "_"

EDIT_KINDS = ["omit", "insert", "sub", "transpose"]


def prev_index(kind: EditKind | None, i, j):
    if kind == "omit":
        return i, j - 1
    elif kind == "insert":
        return i - 1, j
    elif kind == "sub":
        return i - 1, j - 1
    elif kind == "transpose":
        return i - 2, j - 2
    elif kind is None:
        return i - 1, j - 1


@dataclass
class Edit:
    kind: EditKind
    wrong_char: str
    right_char: str

    def as_numerical(self, layout=QWERTY):
        return (
            EDIT_KINDS.index(self.kind),
            layout.get_xy(self.wrong_char),
            layout.get_xy(self.right_char),
        )


def align(str1, str2):
    m, n = (len(str1) + 1, len(str2) + 1)

    str1 += EOS
    str2 += EOS

    d = np.zeros((m, n))

    d[:, 0] = np.arange(m)
    d[0, :] = np.arange(n)

    edit_paths = defaultdict(list)

    for i in range(1, m):
        d[i, 0] = i
        edit_paths[(i, 0)].append(Edit("insert", str1[i - 1], EOS))

    for j in range(1, n):
        d[0, j] = j
        edit_paths[(0, j)].append(Edit("omit", EOS, str2[j - 1]))

    for i in range(1, m):
        for j in range(1, n):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
                sub = None
            else:
                cost = 1
                sub = Edit("sub", str1[i - 1], str2[j - 1])
            edits = [
                Edit("insert", str1[i - 1], str2[j]),
                Edit("omit", str1[i], str2[j - 1]),
                sub,
            ]
            costs = [
                d[i - 1, j] + 1,  # omit
                d[i, j - 1] + 1,  # insert
                d[i - 1, j - 1] + cost,  # substitute
            ]

            if (
                i > 1
                and j > 1
                and str1[i - 2 : i][::-1] == str2[j - 2 : j]
                and str1[i - 2] != str1[i - 1]
            ):
                # transposition possible
                costs.append(d[i - 2, j - 2] + 1)
                edits.append(Edit("transpose", str1[i - 2], str1[i - 1]))

            min_cost = min(costs)
            d[i, j] = min_cost

            for edit, cost in zip(edits, costs):
                if cost == min_cost:
                    edit_paths[(i, j)].append(edit)

    # pd.DataFrame(d.astype(int), index=[str1[:i] for i in range(m)], columns=[str2[:i] for i in range(n)])
    return edit_paths


def all_paths(ij, edit_paths):
    if ij == (0, 0):
        return [[]]

    paths = []
    for edit in edit_paths[ij]:
        edit_kind = None if edit is None else edit.kind
        new_ij = prev_index(edit_kind, *ij)
        paths.extend([path + [edit] for path in all_paths(new_ij, edit_paths)])

    return paths


def total_log_prob(paths, mod: KbdModel, layout=QWERTY):
    probs = []
    for path in paths:
        x0 = []
        x1 = []
        x2 = []
        for edit in path:
            if edit is not None:
                kind, wrong, right = edit.as_numerical(layout)
                x0.append(kind)
                x1.append(list(wrong))
                x2.append(list(right))

        x0 = ops.array(x0)
        x1 = ops.array(x1)
        x2 = ops.array(x2)
        if sum(x0.shape) > 0:
            path_probs = mod.log_prob(x0, x1, x2)
            probs.append(ops.sum(path_probs))
        else:
            probs.append(0)

    return ops.logsumexp(ops.array(probs), axis=0).item()


def kbd_log_prob(str1, str2, mod):
    edit_paths = align(str1, str2)

    if edit_paths:
        paths = all_paths(max(edit_paths.keys()), edit_paths)

        return total_log_prob(paths, mod)
    else:
        return -100


if __name__ == "__main__":
    import pandas as pd
    from scipy.special import softmax
    import keras
    from critic.kbd_model import KbdModel

    mod = KbdModel()
    mod.load_weights("models/kbd_model.weights.h5")

    options = [
        "tere",
        "tear",
        "tree",
        "terr",
        "ere",
        "tee",
        "terse",
        "there",
        "sere",
        "tire",
        "tare",
        "tern",
        "tore",
        "term",
        "mere",
        "here",
        "were",
    ]

    probs = pd.Series(
        [kbd_log_prob(option, "tere", mod) for option in options], options
    )

    probs.loc[:] = softmax(probs)
    print(probs.sort_values(ascending=False))
