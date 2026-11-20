# tests/refusal_metrics.py
"""Refusal precision / recall / F1 from an eval CSV."""
import json
import sys
from pathlib import Path

from tests.eval_report import load_results
from tests.run_evals import REFUSAL_PHRASES


class RefusalMetrics:
    """Treats 'should refuse' as the positive class.

    Recall    = of questions that SHOULD be refused, how many were.
    Precision = of the refusals rostee gave, how many were deserved.
    """

    def __init__(self, csv_paths: list[Path], evals_path: Path) -> None:
        cases = {c["id"]: c for c in json.loads(evals_path.read_text(encoding="utf-8"))}
        # A server error is not a refusal and not an answer, so it must not be scored.
        rows, self._failed = load_results(csv_paths)
        self._rows: list[tuple[bool, bool]] = []
        for row in rows:
            case = cases[row["id"]]
            # Chitchat and 'correct the premise' traps are neither answer nor refusal cases.
            if case["kind"] == "trap" and case.get("trap") != "refuse":
                continue
            should = case.get("trap") == "refuse"
            did = any(p in row["answer"].lower() for p in REFUSAL_PHRASES)
            self._rows.append((should, did))

    def print_summary(self) -> None:
        if self._failed:
            print(f"Not graded (server error): {', '.join(self._failed)}")
        tp = sum(1 for s, d in self._rows if s and d)
        fn = sum(1 for s, d in self._rows if s and not d)
        fp = sum(1 for s, d in self._rows if not s and d)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"Correct refusals: {tp}  Made-up answers: {fn}  Wrongly refused: {fp}")
        print(f"Precision {precision:.2f}  Recall {recall:.2f}  F1 {f1:.2f}")


if __name__ == "__main__":
    RefusalMetrics([Path(p) for p in sys.argv[1:]], Path("tests/evals.json")).print_summary()