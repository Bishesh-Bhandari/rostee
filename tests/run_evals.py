"""Sends every eval question to a running rostee server and grades what it can."""
import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ASK_PATH = "/ask"   # your endpoint name; check /docs if yours differs
PAGE_OFFSET = 0     # set to 1 if rostee shows pages counted from 0
# Only citations from this file count as a page hit: other uploads reuse page numbers.
EVAL_DOC = "Cloud Computing and Virtulization Unit 1 Notes.pdf"
# Must match how your generator words a refusal; copy from a real refusal.
REFUSAL_PHRASES = ("couldn't find", "could not find", "not in the", "no information",
                   "not covered", "doesn't mention", "does not mention", "not mentioned")


@dataclass(frozen=True)
class EvalCase:
    """One question plus its answer key."""
    id: str
    level: str
    kind: str                       # mcq | written | trap | image
    question: str
    answer: str = ""                # mcq letter
    keywords: tuple[str, ...] = ()  # ALL must appear in the answer
    any_of: tuple[str, ...] = ()    # at least ONE must appear
    pages: tuple[int, ...] = ()     # ANY of these cited counts as a page hit
    trap: str = ""                  # refuse | correct | no_sources

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        return cls(d["id"], d.get("level", "T"), d["kind"], d["question"],
                   d.get("answer", ""), tuple(d.get("keywords", ())),
                   tuple(d.get("any_of", ())), tuple(d.get("pages", ())), d.get("trap", ""))


@dataclass(frozen=True)
class Reply:
    """What the server sent back, plus how long it took."""
    text: str
    pages: tuple[int, ...]
    seconds: float
    error: str = ""
    source_count: int = 0           # all sources, from any file


class RosteeClient:
    """Asks one question at a time over HTTP."""

    def __init__(self, base_url: str, timeout: float = 90.0) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def ask(self, question: str) -> Reply:
        start = time.perf_counter()
        try:
            resp = self._http.post(ASK_PATH, json={"question": question})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return Reply("", (), time.perf_counter() - start, str(exc))
        data = resp.json()
        sources = data.get("sources", [])
        pages = tuple(sorted({s["page_number"] + PAGE_OFFSET for s in sources
                              if s["file_name"] == EVAL_DOC}))
        # Gemini sometimes writes ’ for ', which would hide a refusal from the phrase check.
        text = data.get("answer", "").replace("’", "'")
        return Reply(text, pages, time.perf_counter() - start, source_count=len(sources))


class EvalGrader:
    """Gives a first-pass grade. Crude on purpose: every fail still gets a human look."""

    def grade(self, case: EvalCase, reply: Reply) -> tuple[int, int | None]:
        """Returns (answer_ok, page_ok). page_ok is None when no page applies."""
        if reply.error:
            return 0, None
        page_ok = int(bool(set(case.pages) & set(reply.pages))) if case.pages else None
        return int(self._answer_ok(case, reply)), page_ok

    @staticmethod
    def _answer_ok(case: EvalCase, reply: Reply) -> bool:
        text = reply.text.lower()
        if case.kind == "mcq":
            m = re.search(r"\b([A-D])[\).:]|answer is \W*([A-D])\b", reply.text)
            return bool(m) and (m.group(1) or m.group(2)) == case.answer
        if case.kind == "trap":
            if case.trap == "no_sources":
                return reply.source_count == 0
            if case.trap == "correct":
                return any(w in text for w in case.any_of)
            return any(p in text for p in REFUSAL_PHRASES)
        return all(k in text for k in case.keywords)


class EvalRunner:
    """Runs cases one by one, with a pause so the free Gemini tier doesn't block us."""

    def __init__(self, client: RosteeClient, grader: EvalGrader, delay: float) -> None:
        self._client, self._grader, self._delay = client, grader, delay

    def run(self, cases: list[EvalCase], out_path: Path) -> None:
        rows = []
        for case in cases:
            question = case.question
            if case.kind == "mcq":
                question += " Answer with the option letter first."
            reply = self._client.ask(question)
            ok, page_ok = self._grader.grade(case, reply)
            print(f"{case.id:<4} answer={ok} page={page_ok} {reply.seconds:.1f}s {reply.error}")
            rows.append({
                "id": case.id, "level": case.level, "answer_ok": ok,
                "page_ok": "" if page_ok is None else page_ok,
                "made_up": 0,  # NOT measured automatically, review by hand
                "seconds": f"{reply.seconds:.1f}", "notes": reply.error,
                "needs_review": int(ok == 0), "pages_cited": ",".join(map(str, reply.pages)),
                "answer": reply.text.replace("\n", " "),
            })
            time.sleep(self._delay)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="CSV path, e.g. tests/results/run01.csv")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--only", default="", help="comma list of IDs, e.g. M1,W3")
    parser.add_argument("--delay", type=float, default=6.0)
    args = parser.parse_args()
    raw = json.loads(Path(__file__).with_name("evals.json").read_text(encoding="utf-8"))
    cases = [EvalCase.from_dict(d) for d in raw]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    EvalRunner(RosteeClient(args.url), EvalGrader(), args.delay).run(cases, Path(args.out))


if __name__ == "__main__":
    main()