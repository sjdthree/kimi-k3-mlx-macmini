"""Harvest a long-run directory into the KV-cache before/after comparison.

The headline number for this comparison is NOT mean s/token -- that conflates the fixed
cost of streaming 2.8T parameters off disk with the part that grows. It is the SLOPE of
instantaneous s/token against context length.

    master   caches the expanded per-head K/V, 2,949,120 B/token, reallocated and copied
             in full on every decode step -> slope should be strongly positive
    latent   caches the 512-wide latent, 55,296 B/token, chunked growth
             -> slope should be flat within noise

A flat slope is the claim. Mean s/token will also fall if the swap diagnosis is right,
but that is a consequence, not the measurement.

    python tools/kv_cache_report.py runs/polymath --label "master (expanded cache)"
    python tools/kv_cache_report.py runs/polymath-latent --label "latent cache" \
        --compare baseline.json

Writes a markdown block to stdout and, with --json, a machine-readable record so the
"after" run can be compared against the "before" without re-reading either log by hand.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

# Kimi-K3, from config.json. 24 MLA layers: layer 0 plus {3,7,...,91} once
# full_attn_layers is converted from the config's 1-indexed form.
MLA_LAYERS = 24
NH, QHD, VHD = 96, 192, 128
KV_LORA, ROPE = 512, 64

EXPANDED_B_PER_TOK = MLA_LAYERS * (NH * QHD + NH * VHD) * 4      # 2,949,120
LATENT_B_PER_TOK = MLA_LAYERS * (KV_LORA + ROPE) * 4             #    55,296

LINE = re.compile(r"^(\S+)\s+(\d+)/(\d+)\s+tokens")


def parse_log(path: Path) -> list[tuple[dt.datetime, int]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            rows.append((dt.datetime.fromisoformat(m.group(1)), int(m.group(2))))
    return rows


def instantaneous(rows) -> list[tuple[int, float]]:
    """(token index, seconds/token) between consecutive samples."""
    out = []
    for i in range(1, len(rows)):
        dn = rows[i][1] - rows[i - 1][1]
        if dn <= 0:
            continue
        dt_s = (rows[i][0] - rows[i - 1][0]).total_seconds()
        out.append((rows[i][1], dt_s / dn))
    return out


def ols(xs, ys) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx_, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs)
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx_


def analyse(run: Path, prompt_tokens: int) -> dict:
    rows = parse_log(run / "job.log")
    if len(rows) < 3:
        raise SystemExit(f"{run}/job.log has too few progress samples to analyse")
    inst = instantaneous(rows)
    prog = json.loads((run / "progress.json").read_text())

    xs = [t for t, _ in inst]
    ys = [s for _, s in inst]
    slope, intercept = ols(xs, ys)

    # The master curve is NOT a straight line: it climbs steeply while the cache grows
    # into the memory the machine had spare, then plateaus once the system reaches a
    # paging equilibrium. A single slope through that bend flatters the early segment and
    # understates the plateau, so report both thirds and let the shape show.
    third = max(2, len(inst) // 3)
    early_slope, _ = ols(xs[:third], ys[:third])
    late_slope, _ = ols(xs[-third:], ys[-third:])

    k = max(1, len(inst) // 10)
    early = sum(y for _, y in inst[:k]) / k
    late = sum(y for _, y in inst[-k:]) / k

    n_tok = prog.get("tokens_generated_total", rows[-1][1])
    ctx = prompt_tokens + n_tok
    return {
        "run": str(run),
        "state": prog.get("state"),
        "prompt_file": prog.get("prompt_file"),
        "tokens_generated": n_tok,
        "prompt_tokens": prompt_tokens,
        "final_context": ctx,
        "elapsed_s": prog.get("elapsed_s"),
        "mean_s_per_token": prog.get("seconds_per_token"),
        "ttft_s": prog.get("ttft_s"),
        "samples": len(inst),
        "first_decile_s_per_token": round(early, 2),
        "last_decile_s_per_token": round(late, 2),
        "degradation_ratio": round(late / early, 3) if early else None,
        # the headline: seconds/token added per 1000 tokens of context
        "slope_s_per_token_per_1k_ctx": round(slope * 1000, 3),
        "slope_first_third_per_1k_ctx": round(early_slope * 1000, 3),
        "slope_last_third_per_1k_ctx": round(late_slope * 1000, 3),
        "intercept_s_per_token": round(intercept, 2),
        "mla_cache_bytes_at_end": {
            "expanded": ctx * EXPANDED_B_PER_TOK,
            "latent": ctx * LATENT_B_PER_TOK,
        },
        "curve": [{"token": t, "s_per_token": round(s, 2)} for t, s in inst],
    }


def fmt_gib(n: int) -> str:
    return f"{n / 2 ** 30:.2f} GiB"


def render(a: dict, label: str, base: dict | None) -> str:
    L = []
    W = "| {:<34} | {:>18} |".format
    L.append(f"### {label}")
    L.append("")
    L.append(f"`{a['run']}` — prompt `{a['prompt_file']}`, state **{a['state']}**, "
             f"{a['samples']} progress samples")
    L.append("")
    if base:
        L.append("| metric | before | after | change |")
        L.append("|---|---:|---:|---:|")

        def row(name, key, unit="", better="lower"):
            b, x = base.get(key), a.get(key)
            if b in (None, 0) or x is None:
                L.append(f"| {name} | {b} | {x} | — |")
                return
            d = (x - b) / b * 100
            if abs(d) < 0.05:
                arrow = "–"
            else:
                arrow = "✅" if ((d < 0) == (better == "lower")) else "⚠️"
            L.append(f"| {name} | {b}{unit} | {x}{unit} | {d:+.1f}% {arrow} |")

        row("mean s/token", "mean_s_per_token")
        row("first-decile s/token", "first_decile_s_per_token")
        row("last-decile s/token", "last_decile_s_per_token")
        row("degradation (last/first)", "degradation_ratio")
        row("**slope, s/tok per 1k ctx**", "slope_s_per_token_per_1k_ctx")
        row("slope, first third", "slope_first_third_per_1k_ctx")
        row("slope, last third", "slope_last_third_per_1k_ctx")
        L.append(f"| MLA cache at final context | "
                 f"{fmt_gib(base['mla_cache_bytes_at_end']['expanded'])} | "
                 f"{fmt_gib(a['mla_cache_bytes_at_end']['latent'])} | "
                 f"{base['mla_cache_bytes_at_end']['expanded'] / max(a['mla_cache_bytes_at_end']['latent'], 1):.1f}× ✅ |")
    else:
        L.append(W("metric", "value"))
        L.append("|" + "-" * 36 + "|" + "-" * 20 + "|")
        L.append(W("tokens generated", a["tokens_generated"]))
        L.append(W("final context (incl. prompt)", a["final_context"]))
        L.append(W("time to first token", f"{a['ttft_s']:.0f} s"))
        L.append(W("mean s/token", a["mean_s_per_token"]))
        L.append(W("first-decile s/token", a["first_decile_s_per_token"]))
        L.append(W("last-decile s/token", a["last_decile_s_per_token"]))
        L.append(W("degradation (last/first)", f"{a['degradation_ratio']}x"))
        L.append(W("slope, s/tok per 1k ctx", a["slope_s_per_token_per_1k_ctx"]))
        L.append(W("  slope, first third", a["slope_first_third_per_1k_ctx"]))
        L.append(W("  slope, last third", a["slope_last_third_per_1k_ctx"]))
        L.append(W("MLA cache, expanded", fmt_gib(a["mla_cache_bytes_at_end"]["expanded"])))
        L.append(W("MLA cache, latent", fmt_gib(a["mla_cache_bytes_at_end"]["latent"])))

    L.append("")
    L.append("Instantaneous s/token by context length:")
    L.append("")
    L.append("| token | s/token |")
    L.append("|---:|---:|")
    step = max(1, len(a["curve"]) // 12)
    for p in a["curve"][::step]:
        L.append(f"| {p['token']} | {p['s_per_token']} |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--label", default="run")
    ap.add_argument("--prompt-tokens", type=int, default=339,
                    help="prompt length in tokens; only affects the cache-size column. "
                         "339 is the estimate for 04_polymath_multi_expert.md.")
    ap.add_argument("--json", type=Path, help="write the machine-readable record here")
    ap.add_argument("--compare", type=Path, help="a --json record to diff against")
    a = ap.parse_args()

    rec = analyse(a.run, a.prompt_tokens)
    base = json.loads(a.compare.read_text()) if a.compare else None
    print(render(rec, a.label, base))
    if a.json:
        a.json.write_text(json.dumps(rec, indent=2))
        print(f"\n<!-- record written to {a.json} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
