#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tiny ASCII-table formatter. Properly column-pads cells.

Two ways to use:

1. Library:

    from scripts.asciitable import fmt_table
    print(fmt_table(headers=["type", "L0", "L1-3"],
                    rows=[["gate", "0.76", "0.65"], ["up", "0.75", "0.64"]]))

2. CLI (stdin = JSON, stdout = ASCII):

    echo '{"headers":["a","b"],"rows":[[1,2],[3,4]]}' | python scripts/asciitable.py

   The JSON can be either:
     - {"headers": [...], "rows": [[...]]}
     - [{"a":1,"b":2}, {"a":3,"b":4}]    (list of dicts; keys become headers)
"""

from __future__ import annotations

import json
import sys
from typing import Iterable, Sequence


def fmt_table(
    headers: Sequence[object],
    rows: Iterable[Sequence[object]],
    *,
    align: str | Sequence[str] | None = None,
    sep: str = "  ",
) -> str:
    """Return a properly padded ASCII table.

    `align` is a string of "l"/"r"/"c" letters (one per column) or a single
    letter applied to all columns. Default is right-align for numeric-looking
    values, left-align for everything else (decided per column).
    """
    rows_list = [list(r) for r in rows]
    cols = len(headers)
    cells = [[str(c) for c in [h] + [r[i] for r in rows_list]] for i, h in enumerate(headers)]
    widths = [max(len(c) for c in col) for col in cells]

    def _is_num(s: str) -> bool:
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    if align is None:
        # right-align if every data row in this column parses as a number,
        # otherwise left-align
        per_col = []
        for col in cells:
            data = col[1:]
            per_col.append("r" if data and all(_is_num(c) for c in data) else "l")
    elif isinstance(align, str) and len(align) == 1:
        per_col = [align] * cols
    else:
        per_col = list(align)
        assert len(per_col) == cols, "align string must be one letter per column"

    def _pad(s: str, w: int, a: str) -> str:
        if a == "r":
            return s.rjust(w)
        if a == "c":
            return s.center(w)
        return s.ljust(w)

    out = []
    out.append(sep.join(_pad(str(h), widths[i], per_col[i]) for i, h in enumerate(headers)))
    out.append(sep.join("-" * widths[i] for i in range(cols)))
    for r in rows_list:
        out.append(sep.join(_pad(str(r[i]), widths[i], per_col[i]) for i in range(cols)))
    return "\n".join(out)


def _main() -> None:
    raw = sys.stdin.read()
    obj = json.loads(raw)
    if isinstance(obj, list):
        if not obj:
            return
        headers = list(obj[0].keys())
        rows = [[d.get(h, "") for h in headers] for d in obj]
    else:
        headers = obj["headers"]
        rows = obj["rows"]
    print(fmt_table(headers, rows))


if __name__ == "__main__":
    _main()
