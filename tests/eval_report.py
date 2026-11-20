# tests/eval_report.py
"""Turns an eval CSV into the numbers you can put in a README."""
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalRow:
    """One graded question."""
    id: str
    level: str
    answer_ok: int
    page_ok: int | None  # None for traps: there is no right page to cite
    made_up: int
    seconds: float


class EvalReport:
    """Reads a graded CSV and prints accuracy, citation and honesty scores."""

    # The first letter of the ID tells the question type (M1 -> mcq, X3 -> trap).
    _KINDS = {"M": "multiple choice", "W": "written", "X": "trap", "I": "image-only"}

    def __init__(self, csv_path: Path) -> None:
        self._rows = self._load(csv_path)

    def print_summary(self) -> None:
        """Prints overall, per-type and per-level scores."""
        rows = self._rows
        cited = [r for r in rows if r.page_ok is not None]
        print(f"Questions:        {len(rows)}")
        print(f"Answer accuracy:  {self._pct(sum(r.answer_ok for r in rows), len(rows))}")
        print(f"Right page cited: {self._pct(sum(r.page_ok for r in cited), len(cited))}")
        print(f"Made-up answers:  {self._pct(sum(r.made_up for r in rows), len(rows))}")
        print(f"Avg time:         {sum(r.seconds for r in rows) / len(rows):.1f}s")
        self._print_groups("By type", lambda r: self._KINDS.get(r.id[0], "other"))
        self._print_groups("By level", lambda r: r.level)

    def _print_groups(self, title: str, key) -> None:
        groups: dict[str, list[EvalRow]] = defaultdict(list)
        for row in self._rows:
            groups[key(row)].append(row)
        print(f"\n{title}:")
        for name in sorted(groups):
            rows = groups[name]
            print(f"  {name:<16} {self._pct(sum(r.answer_ok for r in rows), len(rows))}")

    @staticmethod
    def _pct(good: int, total: int) -> str:
        return f"{100 * good / total:.0f}% ({good}/{total})" if total else "n/a"

    @staticmethod
    def _load(path: Path) -> list[EvalRow]:
        # Empty page_ok cell means "not applicable", so it must not count as a miss.
        with path.open(newline="", encoding="utf-8") as f:
            return [
                EvalRow(
                    id=r["id"],
                    level=r["level"],
                    answer_ok=int(r["answer_ok"]),
                    page_ok=int(r["page_ok"]) if r["page_ok"] else None,
                    made_up=int(r["made_up"]),
                    seconds=float(r["seconds"] or 0),
                )
                for r in csv.DictReader(f)
            ]


if __name__ == "__main__":
    EvalReport(Path(sys.argv[1])).print_summary()