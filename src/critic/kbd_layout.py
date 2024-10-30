"""
Modeling keyboard layouts.
"""

from dataclasses import dataclass
from logging import warning
from typing import Sequence


@dataclass
class KbdLayout:
    letters: str
    width: int
    height: int

    def __init__(self, layout: Sequence[str]):
        self.width = len(layout[0])
        self.height = len(layout)
        self.letters = "".join(layout).lower()

    def get_xy(self, char):
        if char.lower() not in self.letters:
            # warning(f"{char.lower()} not found, using (0, 0)")
            return (0, 0)
        else:
            i = self.letters.index(char.lower())
            return (i % self.width, self.height - (i // self.width) - 1)


QWERTY = KbdLayout(
    [
        "qwertyuiop",
        "asdfghjkl;",
        "zxcvbnm,./",
    ]
)
