__all__ = ["StatsFormatter",
           "DialogFormatter", "DialogTableFormatter", "DialogMarkdownFormatter",
           "DialogLatexFormatter", "DialogICMLLatexFormatter"]

import importlib
import itertools
from dataclasses import dataclass
from typing import Sequence, ClassVar

import prettytable

from .sugar import TensorMap
from .sugar import pad_iter


def chunk(x, n):
    buf = []
    for e in x:
        if len(buf) >= n:
            yield buf
            buf = []
        buf.append(e)
    if buf:
        yield buf + [None] * max(0, n - len(buf))


@dataclass
class StatsFormatter:
    num_cols: int = 3

    def format(self, stats: TensorMap, title=None) -> str:
        t = prettytable.PrettyTable(list(itertools.chain(
            *[[f"Id{i}", f"K{i}", f"V{i}"]
              for i in range(1, self.num_cols + 1)]
        )))
        if title is not None:
            t.title = title
        for i in range(1, self.num_cols + 1):
            t.align[f"K{i}"] = "l"
            t.align[f"V{i}"] = "r"
        t.float_format = " 4.5"
        for items in chunk(enumerate(stats.items()), self.num_cols):
            row = []
            for item in items:
                if item is None:
                    row.extend(("", "", ""))
                    continue
                idx, (k, v) = item
                row.extend((idx + 1, k, v.item()))
            t.add_row(row)
        return str(t)


@dataclass
class DialogFormatter:

    def format(self, dialog):
        raise NotImplementedError


@dataclass
class DialogTableFormatter(DialogFormatter):
    max_col_len: int = 50

    def format(self, dialog) -> str:
        dialogs = [dialog]
        if not dialogs:
            return ""
        if len(dialogs) == 1:
            columns = ["TURN", "SENT", "SPKR", "GOAL", "STATE"]
        else:
            columns = ["TURN"] + list(itertools.chain(*(
                [f"SENT{i}", f"SPKR{i}", f"GOAL{i}", f"STATE{i}"]
                for i in range(1, len(dialogs) + 1)
            )))
        t = prettytable.PrettyTable(columns)
        t.align = "l"
        max_len = max(map(len, dialogs))
        data = [list(map(str, range(1, max_len + 1)))]
        datasets = importlib.import_module("datasets")
        dummy = datasets.Turn("", "")
        data.extend(itertools.chain(*(
            ([turn.text for turn in pad_iter(dialog.turns, max_len, dummy)],
             [turn.speaker for turn in pad_iter(dialog.turns, max_len, dummy)],
             [turn.goal.to_cam()
              for turn in pad_iter(dialog.turns, max_len, dummy)],
             [turn.state.to_cam()
              for turn in pad_iter(dialog.turns, max_len, dummy)])
            for dialog in dialogs
        )))
        data = [[s if len(s) <= self.max_col_len else
                 s[:self.max_col_len - 3] + "..." for s in d] for d in data]
        rows = zip(*data)
        tuple(map(t.add_row, rows))
        return str(t)


@dataclass
class DialogMarkdownFormatter(DialogFormatter):
    max_col_len: int = 50

    @staticmethod
    def render_mdtable(headers, rows, max_col_width=15) -> str:
        def render_row(cells, cell_widths):
            def pad_cell(c, width):
                c = str(c)
                if len(c) > width:
                    c = c[:width - 2] + ".."
                return c

            return "|".join(pad_cell(cell, cell_width)
                            for cell, cell_width in zip(cells, cell_widths))

        rows = [tuple(map(str, row)) for row in rows]
        num_cols = len(headers)
        col_widths = [max([min(len(row[i]), max_col_width)
                           if i < len(row) else 0 for row in rows] +
                          [len(headers[i])])
                      for i in range(num_cols)]
        rows = [headers, ["-" * width for width in col_widths]] + list(rows)
        return "\n".join(render_row(pad_iter(row, num_cols, ""), col_widths)
                         for row in rows)

    def format(self, dialog):
        """Renders the sequence of dialogs as a markdown table."""
        return self.render_mdtable(
            headers=["Turn", "Sent", "Spkr", "Goal", "State"],
            rows=[(i + 1, turn.text, turn.speaker, turn.goal, turn.state)
                  for i, turn in enumerate(dialog.turns)],
            max_col_width=self.max_col_len
        )


@dataclass
class DialogLatexFormatter(DialogFormatter):
    column_ratios: Sequence[float] = None

    def __post_init__(self):
        self.column_ratios = self.column_ratios or (0.05, 0.10, 0.4, 0.22, 0.22)
        if len(self.column_ratios) != 5:
            raise ValueError(f"the number of elements in column ratios must be "
                             f"5, where each element references turn id, "
                             f"speaker, utterance, goal, and turn act columns")

    @property
    def turn_colwidth(self):
        return rf"{self.column_ratios[0]}\linewidth"

    @property
    def speaker_colwidth(self):
        return rf"{self.column_ratios[1]}\linewidth"

    @property
    def utterance_colwidth(self):
        return rf"{self.column_ratios[2]}\linewidth"

    @property
    def goal_colwidth(self):
        return rf"{self.column_ratios[3]}\linewidth"

    @property
    def act_colwidth(self):
        return rf"{self.column_ratios[4]}\linewidth"

    @classmethod
    def required_packages(cls):
        return "xcolor", "colortbl"

    @staticmethod
    def render_state(state):
        items = [(act, ", ".join(map(str, svs)))
                 for act, svs in state.data.items()]
        items = list(sorted(items, key=lambda x: x[0]))
        return r" \newline ".join(f"{act}({svs})" for act, svs in items)

    def render_turn(self, turn):
        return " & ".join([
            turn.speaker.capitalize(),
            turn.text.capitalize(),
            self.render_state(turn.goal),
            self.render_state(turn.state)
        ])

    def format(self, dialog):
        col_spec = "|".join(rf"p{{{ratio}\linewidth}}"
                            for ratio in self.column_ratios)
        hline = rf"\noalign{{\global\arrayrulewidth=0.25\arrayrulewidth}}" \
            rf"\arrayrulecolor{{lightgray}}\hline" \
            f"\\noalign{{\\global\\arrayrulewidth=4.0\\arrayrulewidth}}" \
            f"\\arrayrulecolor{{black}}\n"
        hline = " " * 8 + hline
        rows = f"\\\\\n{hline}".join(
            f"{' ' * 8}{idx} & {self.render_turn(turn)}"
            for idx, turn in enumerate(dialog.turns))
        return rf"""
\begin{{table}}
    \centering
    \begin{{tabular}}{{{col_spec}}}
        \hline
        \textbf{{Turns}} & \textbf{{Speaker}} & \textbf{{Utterance}} & \textbf{{Goal}} & \textbf{{Turn Act}} \\
        \hline\hline
{rows}\\
        \hline
    \end{{tabular}}
\end{{table}}
        """


@dataclass
class DialogICMLLatexFormatter(DialogFormatter):
    placeholders: ClassVar[dict] = {
        "loc": r"\verb|<location>|",
        "numeric": r"\verb|<numeric>|",
        "someplace": r"\verb|<place>|"
    }

    @classmethod
    def required_packages(cls):
        return "xcolor", "colortbl"

    @staticmethod
    def render_state(state):
        items = [(act, sv) for act, svs in state.data.items() for sv in svs]
        items = list(sorted(items, key=lambda x: x[0]))
        return r" \newline ".join(f"{act}({svs})" for act, svs in items)

    def render_text(self, text):
        return " ".join(self.placeholders.get(w, w) for w in text.split())

    def render_turn(self, turn):
        return " & ".join([
            turn.speaker.capitalize(),
            self.render_text(turn.text).lower(),
            self.render_state(turn.goal),
            self.render_state(turn.state)
        ])

    def format(self, dialog):
        col_spec = r"p{0.02\linewidth}p{0.11\linewidth}" \
                   r">{\raggedright}p{0.30\linewidth}" \
                   r"p{0.24\linewidth}p{0.24\linewidth}"
        hline = r"\weakline"
        hline = " " * 8 + hline
        rows = f"\\\\\n{hline}\n".join(
            f"{' ' * 8}{idx} & {self.render_turn(turn)}"
            for idx, turn in enumerate(dialog.turns))
        return rf"""
\begin{{table}}
    \caption{{}}
    \label{{}}
    \vskip 0.15in
    \small
    \setlength\tabcolsep{{1.5pt}}
    \centering
    \begin{{tabular}}{{{col_spec}}}
        \toprule
        & \textbf{{Speaker}} & \textbf{{Utterance}} & 
          \textbf{{Goal}} & \textbf{{Turn Act}} \\
        \midrule
{rows}\\
        \bottomrule
    \end{{tabular}}
\end{{table}}
        """
