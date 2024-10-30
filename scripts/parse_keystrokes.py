from chex import dataclass as _dataclass
import numpy as np
import pandas as pd
from collections.abc import Sequence
# import seaborn as sns
# import rho_plus as rp
# is_dark = False
# theme, cs = rp.mpl_setup(is_dark)


dataclass = _dataclass(mappable_dataclass=False)


@dataclass
class KeyEvent:
    """Key event: either a press or release."""

    is_press: bool
    letter: str


BKSP = "␈"
OMIT = "¬"
SPACER = "␣"


@dataclass
class TextState:
    """State of typing, including SHIFT."""

    shift_down: bool = False
    typed_letters: str = ""
    all_letters: str = ""


def type_key(event: KeyEvent, state: TextState) -> TextState:
    if event.letter == "SHIFT":
        return state.replace(shift_down=event.is_press)
    elif not event.is_press:
        # only pressing a key matters, unless it's shift
        return state
    elif event.letter == "BKSP":
        return state.replace(
            typed_letters=state.typed_letters[:-1], all_letters=state.all_letters + BKSP
        )
    elif len(event.letter) == 1:  # single key press
        if state.shift_down:
            letter = event.letter.upper()
        else:
            letter = event.letter.lower()
        return state.replace(
            typed_letters=state.typed_letters + letter,
            all_letters=state.all_letters + letter,
        )
    elif event.letter == "CTRL":
        # ignore, I suppose?
        return state
    else:
        raise NotImplementedError(
            f"Support for {event.letter} not available right now."
        )


def parse_subj(subj_id):
    # with ZipFile("/home/nicholas/Downloads/Keystrokes.zip") as z:
    #     with z.open(f"Keystrokes/files/{subj_id}_keystrokes.txt") as f:
    #         subj_data = pd.read_csv(f, sep="\t", encoding="latin")

    subj_data = pd.read_csv(
        f"~/Downloads/Keystrokes/files/{subj_id}_keystrokes.txt",
        sep="\t",
        encoding="latin",
    )

    # subj_data = pd.read_csv(f"data/keystrokes/{subj_id}_keystrokes.txt", sep="\t", encoding='latin')
    # for col in ("PRESS_TIME", "RELEASE_TIME"):
    #     subj_data[col] = pd.to_datetime(subj_data[col], unit="ms")

    events = []
    for i, row in subj_data.iterrows():
        events.append((row["PRESS_TIME"], i, KeyEvent(True, row["LETTER"])))
        events.append((row["RELEASE_TIME"], i, KeyEvent(False, row["LETTER"])))

    events = sorted(events, key=lambda x: x[0])
    state = TextState()

    subj_data["text"] = ""
    subj_data["input_stream"] = ""

    flagged_sents = set()
    for _time, i, event in events:
        try:
            state = type_key(event, state)
        except NotImplementedError:
            flagged_sents.add(subj_data.loc[i, "SENTENCE"])
        if event.is_press:
            ix = list(subj_data.index).index(i)
            if (
                ix == len(subj_data.index) - 1
                or len(pd.unique(subj_data["SENTENCE"].iloc[ix : ix + 2])) == 2
            ):
                if subj_data.loc[i, "SENTENCE"] not in flagged_sents:
                    subj_data.loc[i, "text"] = state.typed_letters
                    subj_data.loc[i, "input_stream"] = state.all_letters
                state = TextState()

    subj_data = subj_data.query('input_stream != ""')
    return subj_data


# The below algorithm is copied nearly line-for-line from the pseudocode in
# https://faculty.washington.edu/wobbrock/pubs/tochi-06.pdf

# The only novel addition is adding in extra backspaces if they make the alignment better. Some
# participants clearly show that they're holding down backspace and letting it register multiple
# times, which isn't shown in the data.


@dataclass
class InputStream:
    text: str
    flags: Sequence[bool]
    positions: Sequence[int] = ()

    def __str__(self):
        out = []
        for letter, flag in zip(self.text, self.flags):
            if flag:
                out.append("\u0304" + letter)
            else:
                out.append(letter)

        return "".join(out)


def flag_stream(input_stream: str):
    flags = np.zeros(len(input_stream), dtype=np.bool_)
    count = 0
    for i, letter in reversed(list(enumerate(input_stream))):
        if letter == BKSP:
            count += 1
        else:
            if count == 0:
                flags[i] = True
            else:
                count -= 1
    return InputStream(input_stream, flags)


def msd_matrix(p: str, t: str):
    d = np.zeros((len(p) + 1, len(t) + 1), dtype=np.uint16)

    d[:, 0] = np.arange(len(p) + 1)
    d[0, :] = np.arange(len(t) + 1)

    for i in range(1, len(p) + 1):
        for j in range(1, len(t) + 1):
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + (p[i - 1] != t[j - 1]),
            )

    return d


# def align(p, t, d, x, y, p1, t1, alignments):
#     if x == y == 0:
#         alignments.append((p1, t1))
#         return alignments

#     if x > 0 and y > 0:
#         if d[x, y] == d[x - 1, y - 1] and p[x - 1] == t[y - 1]:
#             align(p, t, d, x - 1, y - 1, p[x - 1] + p1, t[y - 1] + t1, alignments)
#         if d[x, y] == d[x - 1, y - 1] + 1:
#             align(p, t, d, x - 1, y - 1, p[x - 1] + p1, t[y - 1] + t1, alignments)

#     if x > 0 and d[x, y] == d[x - 1, y] + 1:
#         align(p, t, d, x - 1, y, p[x - 1] + p1, OMIT + t1, alignments)

#     if y > 0 and d[x, y] == d[x, y - 1] + 1:
#         align(p, t, d, x, y - 1, OMIT + p1, t[y - 1] + t1, alignments)

#     return alignments


def align(p, t, d, x, y, p1, t1, alignments):
    stack = [(p, t, d, x, y, p1, t1)]

    while stack:
        p, t, d, x, y, p1, t1 = stack.pop()
        if x == y == 0:
            alignments.append((p1, t1))
            return alignments

        if x > 0 and y > 0:
            if (d[x, y] == d[x - 1, y - 1] and p[x - 1] == t[y - 1]) or d[x, y] == d[
                x - 1, y - 1
            ] + 1:
                stack.append((p, t, d, x - 1, y - 1, p[x - 1] + p1, t[y - 1] + t1))

        if x > 0 and d[x, y] == d[x - 1, y] + 1:
            stack.append((p, t, d, x - 1, y, p[x - 1] + p1, OMIT + t1))

        if y > 0 and d[x, y] == d[x, y - 1] + 1:
            stack.append((p, t, d, x, y - 1, OMIT + p1, t[y - 1] + t1))

    return alignments


def msd_align(p, t, d=None):
    if d is None:
        d = msd_matrix(p, t)

    alignments = []
    align(p, t, d, len(p), len(t), "", "", alignments)
    return alignments, d


def stream_align(input_stream: InputStream, alignments):
    triplets = []
    for p, t in alignments:
        stream = list(input_stream.text)
        flags = list(input_stream.flags)
        for i in range(max(len(t), len(stream))):
            if i < len(t) and t[i] == OMIT:
                stream.insert(i, SPACER)
                flags.insert(i, True)
            elif not flags[i]:
                p = p[:i] + SPACER + p[i:]
                t = t[:i] + SPACER + t[i:]
        triplets.append((p, t, InputStream("".join(stream), flags)))

    return triplets


def assign_position_values(triplets):
    for p, t, i_s in triplets:
        i_s.positions = []
        pos = 0
        for i in range(len(i_s.text)):
            if i_s.flags[i]:
                i_s.positions.append(0)
                pos = 0
            else:
                if i_s.text[i] == BKSP and pos > 0:
                    pos -= 1

                i_s.positions.append(pos)
                if i_s.text[i] != BKSP:
                    pos += 1


@dataclass
class Error:
    kind: str
    corrected: bool
    i: int
    j: int | None = None


def look_ahead(s, start, count, predicate):
    index = start
    while 0 <= index < len(s) and not predicate(s[index]):
        index += 1

    while count > 0 and index < len(s):
        index += 1
        if index == len(s):
            break
        elif predicate(s[index]):
            count -= 1

    return index


def look_behind(s, start, count, predicate):
    index = start
    while 0 <= index < len(s) and not predicate(s[index]):
        index -= 1

    while count > 0 and index >= 0:
        index -= 1
        if index < 0:
            break
        elif predicate(s[index]):
            count -= 1

    return index


def is_letter(s):
    return s != BKSP


def determine_errors(triplets):
    all_errors = {}
    for p, t, i_s in triplets:
        errors = []
        a = 0
        for b in range(len(i_s.text)):
            if b < len(t) and t[b] == OMIT:
                errors.append(Error("omission", False, b))
            elif b == len(i_s.text) - 1 or i_s.flags[b]:
                omissions = set()
                insertions = set()
                for i in range(a, b):
                    is_i = i_s.text[i]
                    v = i_s.positions[i]
                    if is_i == BKSP:
                        omissions.discard(v)
                        insertions.discard(v)
                    elif is_i != "_":
                        target = look_ahead(
                            p, b, v + len(omissions) - len(insertions), is_letter
                        )
                        # skipping nonrecognized case
                        next_p = look_ahead(p, target, 1, is_letter)
                        prev_p = look_behind(p, target, 1, is_letter)
                        next_is = look_ahead(i_s.text, i, 1, lambda s: s != SPACER)
                        prev_is = look_behind(i_s.text, i, 1, lambda s: s != SPACER)
                        if target < len(p) and is_i == p[target]:
                            # corrected no error
                            pass
                        elif (
                            target >= len(p)
                            or i_s.text[next_is] == p[target]
                            or (i_s.text[prev_is] == is_i == p[prev_p])
                        ):
                            errors.append(Error("insertion", True, i))
                            insertions.add(v)
                        elif (
                            0 <= next_p < len(p)
                            and is_i == p[next_p]
                            and target < len(t)
                            and is_letter(t[target])
                        ):
                            errors.append(Error("omission", True, target))
                            # corrected no error IS[i]
                            omissions.add(v)
                        else:
                            errors.append(Error("substitute", True, target, i))

                if b < len(p):
                    if p[b] == OMIT:
                        errors.append(Error("insertion", False, b))
                    elif p[b] != t[b]:
                        errors.append(Error("substitute", False, b, b))
                    elif p[b] != SPACER:
                        # uncorrected no error
                        pass
                # skipping norec-insertion
                a = b + 1
        all_errors[(p, t, i_s.text)] = errors

    return all_errors


def print_errors(all_errors):
    for (p, t, ist), errors in all_errors.items():
        print(p, t, ist, "", sep="\n")
        for error in errors:
            if error.corrected:
                print("  corrected", error.kind, end="\t")
                i = error.i
                j = error.j
                if error.kind == "insertion":
                    print(f"{ist[i]}={i}")
                elif error.kind == "substitute":
                    print(f"({p[i]}={i} -> {ist[j]}={j})")
                elif error.kind == "omission":
                    print(f"{p[i]}={i}")
            else:
                print("uncorrected", error.kind, end="\t")
                i = error.i
                j = error.j
                if error.kind == "insertion":
                    print(f"{t[i]}={i}")
                elif error.kind == "substitute":
                    print(f"({p[i]}={i} -> {t[j]}={j})")
                elif error.kind == "omission":
                    print(f"{p[i]}={i}")


# all_errs = determine_errors(triplets)
# print_errors(all_errs)


def all_errors(p, t, ist):
    alignments, d = msd_align(p, t)
    stream = flag_stream(ist)
    triplets = stream_align(stream, alignments)
    assign_position_values(triplets)
    all_errors = determine_errors(triplets)
    return all_errors


def min_err(all_errors):
    return min(
        len([e for e in errs if not e.corrected]) for errs in all_errors.values()
    )


def to_text(ist):
    s = []
    for c in ist:
        if c == BKSP:
            if s:
                s.pop()
            else:
                pass
        else:
            s.append(c)

    return "".join(s)


def find_best_alignments(p, t, ist, skip=0):
    if BKSP not in ist:
        return all_errors(p, t, ist), ist
    else:
        i = 0
        skipped = skip
        for i, l in enumerate(ist):
            if l == BKSP and skipped == 0:
                # test if adding more backspaces would make alignment better
                best_ist = ist
                best_align = all_errors(p, t, ist)
                best_err = min_err(best_align)
                best_bksp = 1
                num_bksp = 2
                new_ist = best_ist
                while i - num_bksp >= 0 and ist[i + 1 - num_bksp] != BKSP:
                    new_ist = new_ist[:i] + BKSP + new_ist[i:]
                    new_t = to_text(new_ist)
                    if new_t is None:
                        break
                    next_align = all_errors(p, new_t, new_ist)
                    next_err = min_err(next_align)
                    # print(new_ist, '-', new_t, '-', next_align, next_err)
                    if next_err < best_err:
                        best_align = next_align
                        best_err = next_err
                        best_ist = new_ist
                        best_bksp = num_bksp
                    elif next_err > best_err:
                        break

                    num_bksp += 1

                return find_best_alignments(
                    p, to_text(best_ist), best_ist, skip + best_bksp
                )
            elif l == BKSP:
                skipped -= 1
            i += 1

        return all_errors(p, t, ist), ist


def context(p, i):
    """Gets the text up to index i and the subsequent two letters, removing spacers."""
    return p[: i + 1].replace(SPACER, "") + (p[i + 1 :].replace(SPACER, "") + "__")[:2]


def collect_errors(last_rows):
    from tqdm import tqdm

    err_df = []

    for sent, text, i_s, part_id in tqdm(last_rows):
        try:
            errors, curr_ist = find_best_alignments(sent, text, i_s)
        except Exception as e:
            # print(e)
            continue
        for (p, t, ist), errs in errors.items():
            for error in errs:
                i = error.i
                j = error.j

                ctx = context(p, i)

                if error.corrected:
                    user_text = ist
                else:
                    user_text = t
                if error.kind == "insertion":
                    wrong_char = user_text[i]
                    right_char = ctx[-2]
                elif error.kind == "substitute":
                    wrong_char = user_text[j]
                    right_char = p[i]
                elif error.kind == "omission":
                    wrong_char = ctx[-2]
                    right_char = p[i]

                row = (
                    error.kind,
                    error.corrected,
                    part_id,
                    1 / len(errors),
                    p,
                    t,
                    ist,
                    i,
                    j,
                    ctx,
                    wrong_char,
                    right_char,
                )

                err_df.append(row)

    return err_df


def get_rows():
    import pandas as pd
    from tqdm import tqdm

    metadata = pd.read_csv("data/keystrokes/metadata_participants.txt", sep="\t")

    # ## Participant Selection
    #
    # For now, this project has a specific scope: people who are taking a USC CS class, and at least for
    # now people who are using laptops or desktops. That means we have a few things to filter out:
    #
    # - Dealing with different keyboard layouts is fine, so long as we properly account for it.
    # - I think it makes the most sense to target an audience that is at least typing with most of their
    #   fingers: "hunt-and-peck" is just a very different kind of typing with its own potential mistakes,
    #   and I would guess the large majority of people in CS have at least learned to type reasonably
    #   well.
    # - We obviously want to look at the errors, but it doesn't make sense to consider people who have an
    #   abnormally large amount of errors: we'll filter out the outliers.

    # Seeing the "10+" option has piqued my interest: how do people with hexadactyly type? Am I missing
    # out?
    #
    # We'll filter to 7-8 and above: we have plenty of data there.

    filtered = metadata.query(
        'FINGERS == "7-8" or FINGERS == "9-10" or FINGERS == "10+"'
    )

    # We could add support for other keyboard layouts here, but QWERTY is good enough and avoids any
    # potential confusion.
    filtered = metadata.query('LAYOUT == "qwerty"')

    # We'll filter to "full" and "laptop".

    filtered = filtered.query('KEYBOARD_TYPE == "full" or KEYBOARD_TYPE == "laptop"')

    # This distribution of typing speed looks quite natural, so we won't do any filtering on it.

    # The errors per character does have a few outliers. We'll tentatively filtered out anyone who has
    # more than a 40% error rate.

    filtered = filtered.query("ECPC <= 0.4")

    input_rows = []
    for subj_id in tqdm(filtered["PARTICIPANT_ID"].iloc[::129]):
        try:
            subj_data = parse_subj(subj_id)
        except Exception as e:
            # print(e)
            continue

        for i, row in subj_data.iterrows():
            # print(i)
            input_rows.append(
                (
                    row["SENTENCE"],
                    row["text"],
                    row["input_stream"],
                    row["PARTICIPANT_ID"],
                )
            )

    return input_rows


def collect_rows(row_errs):
    import pandas as pd

    df = pd.DataFrame(
        row_errs,
        columns=(
            "kind",
            "corrected",
            "participant",
            "weight",
            "p",
            "t",
            "ist",
            "i",
            "j",
            "context",
            "wrong_char",
            "right_char",
        ),
    )

    for cat_col in ("participant", "p", "t", "ist"):
        df[cat_col] = df[cat_col].astype("category")

    df["kind"] = pd.Categorical(
        df["kind"], categories=("insertion", "omission", "substitute", "transpose")
    )
    df = df.reset_index(drop=True)

    df["transpose_1"] = False
    df["transpose_2"] = False
    for i in range(len(df.index) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        if (
            (row1["kind"] == "insertion" and row2["kind"] == "omission")
            and (row1["wrong_char"] == row2["right_char"])
            and (("_" + row2["context"])[-3] == row1["right_char"])
        ):
            df.loc[row1.name, "transpose_1"] = True
            df.loc[row2.name, "transpose_2"] = True
            # display(df.iloc[i:i+2])

        if (row1["kind"] == "omission" and row2["kind"] == "insertion") and (
            row2["wrong_char"] == row1["right_char"]
        ):
            df.loc[row1.name, "transpose_1"] = True
            df.loc[row2.name, "transpose_2"] = True

        if row1["kind"] == "insertion" and row1["context"][-1] == row1["wrong_char"]:
            df.loc[row1.name, "transpose_1"] = True
            # display(df.iloc[i:i+1])

    df.loc[df["transpose_1"], "kind"] = "transpose"
    df = df.query("not transpose_2").drop(columns=["transpose_1", "transpose_2"])

    df.to_feather("precomputed/all_typos.feather")


def main():
    rows = get_rows()
    row_errs = collect_errors(rows)
    return collect_rows(row_errs)


if __name__ == "__main__":
    main()
