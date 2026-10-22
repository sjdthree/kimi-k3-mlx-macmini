"""Collect the finished local runs into ../local-results/, in the cloud reference's layout.

Reads runs/demo-local/<case>/ (written by tools/run_chat_job.py via run.sh) and writes
results.md (each case's answer, byte-for-byte as decoded, with its run metadata) and
summary.json. The reasoning/answer token split uses the same channel parser as the job,
so it needs the checkpoint's tokenizer:

    python demos/private-document-analysis/local/collect_results.py --model "$K3_CKPT"

The raw run files (tokens.jsonl, job.log, reasoning.md, ...) are committed separately
under runs/demo-local/.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEMO = ROOT / "demos" / "private-document-analysis"
sys.path.insert(0, str(ROOT))

# Same order and numbering as cloud-reference/.
CASES = [
    ("semiconductor", "semiconductor_root_cause", "01_semiconductor_confidential_excursion.pdf"),
    ("medical_ai", "medical_ai_adverse_event", "02_medical_ai_confidential_incident.pdf"),
    ("rd_tradesecret", "rd_tradesecret_invention_strategy", "03_rd_tradesecret_invention_strategy.pdf"),
]
START = re.compile(r"^(\S+) prompt \d+ tokens, (\d+) generated so far")
STEP = re.compile(r"^(\S+) \d+ tokens, ")


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def sessions(log: list[str]) -> list[dict]:
    """One entry per job start that went on to decode: when it started, what it resumed
    from, and the time of its last progress line."""
    out: list[dict] = []
    for line in log:
        if m := START.match(line):
            out.append({"start": m[1], "resumed_from": int(m[2]), "last": m[1]})
        elif out and (m := STEP.match(line) or re.match(r"^(\S+) done: ", line)):
            out[-1]["last"] = m[1]
    return [s for s in out if s["last"] != s["start"]]


def case_stats(tok, name: str) -> dict:
    from air_llm.models.kimi_k3.chat_format import ChannelParser

    d = ROOT / "runs" / "demo-local" / name
    recs = [json.loads(line) for line in (d / "tokens.jsonl").open() if line.strip()]
    meta = json.loads((d / "meta.json").read_text())
    prog = json.loads((d / "progress.json").read_text())
    parser = ChannelParser(tok, thinking=True)
    n = {"reasoning": 0, "content": 0, None: 0}
    for r in recs:
        n[parser.feed(r["id"])] += 1
    secs = [r["s"] for r in recs]
    decode = secs[1:]  # the first token's time is the end of prefill, not a decode step
    runs = sessions((d / "job.log").read_text().splitlines())
    gaps = [(a["last"], b["start"]) for a, b in zip(runs, runs[1:])]
    return {
        "status": prog["status"],
        "finish": "end_of_message" if prog["status"] == "done" else None,
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": len(recs),
        "reasoning_tokens": n["reasoning"],
        "answer_tokens": n["content"],
        "format_tokens": n[None],
        "decode_seconds_per_token_median": round(statistics.median(decode), 2),
        "decode_seconds_per_token_max": max(decode),
        "generation_hours": round(sum(secs) / 3600, 1),
        "peak_mlx_gib": max(r["peak_gib"] for r in recs),
        "peak_mlx_gib_decode": max(r["peak_gib"] for r in recs[2:]),
        "started": meta["started"],
        "finished": prog["updated"],
        "interruptions": [{"stopped": a, "restarted": b,
                           "hours_down": round((ts(b) - ts(a)).total_seconds() / 3600, 1)}
                          for a, b in gaps],
        "settings": {k: meta[k] for k in ("temperature", "top_p", "seed", "thinking_effort",
                                          "max_tokens", "resident_gb", "mode")},
        "messages": meta["messages"],
        "messages_sha256": meta["messages_sha256"],
        "model_config_sha256": meta["model_config_sha256"],
        "commits": meta["commits"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint directory (for the tokenizer)")
    a = ap.parse_args()
    from air_llm.models.kimi_k3.tokenizer import load as load_tok

    tok = load_tok(Path(a.model))
    out = DEMO / "local-results"
    out.mkdir(exist_ok=True)

    results, md = [], [
        "# Kimi-K3 Private Document Evaluation — local run", "",
        "Model: full Kimi-K3 checkpoint, run by this repository on a 64 GiB M4 Pro Mac mini",
        "Inputs: [`local/inputs/`](../local/inputs/) (PDF text plus the question; one user message, no system prompt)",
        "Maximum completion tokens: 20,000", "", "---", "",
    ]
    for i, (name, label, pdf) in enumerate(CASES, 1):
        s = case_stats(tok, name)
        results.append({"number": i, "name": label, "source_document": pdf, **s})
        answer = (ROOT / "runs" / "demo-local" / name / "answer.md").read_text()
        question = (DEMO / "prompts" / f"{name}.txt").read_text().strip()
        md += [f"# {i}. {label}", "", "## Source Document", "", f"`{pdf}`", "",
               "## Question", "", question, "", "## Kimi-K3 Answer", "", answer.rstrip(), "",
               "## Run Metadata", "", "```json", json.dumps(s, indent=2), "```", "", "---", ""]

    summary = {
        "model": "Kimi-K3 (local, this repository)",
        "machine": "Mac mini M4 Pro, 64 GiB unified memory; checkpoint on USB4 NVMe",
        "max_completion_tokens": 20000,
        "cases_completed": sum(r["status"] == "done" for r in results),
        "results": results,
    }
    (out / "results.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("name", "completion_tokens", "reasoning_tokens",
                                         "generation_hours", "interruptions")} for r in results],
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
