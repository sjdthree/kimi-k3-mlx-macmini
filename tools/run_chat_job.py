"""Run one multi-day chat generation in the K3 chat format, resumable exactly.

    python tools/run_chat_job.py --model "$K3_CKPT" --manifest "$K3_MANIFEST" \
        --cache "$K3_CACHE" --resident-gb 8 \
        --messages demos/private-document-analysis/inputs/semiconductor.json \
        --out runs/demo-local/semiconductor --max-tokens 12000

Check on it from anywhere:

    cat runs/demo-local/semiconductor/progress.json
    tail -f runs/demo-local/semiconductor/reasoning.md

Unlike `run_long_job.py`, this loads the model in-process instead of going through the
server, so it can resume a generation where it stopped rather than replaying text:

* every sampled token is appended to `tokens.jsonl` and fsynced before the next step;
* the recurrent state and caches are saved every `--save-every` tokens (atomically);
* token i is sampled with seed `--seed + i`, so a resumed run draws the same numbers.

On restart with the same `--out`, the saved state is loaded and the tokens generated
after it are fed back one at a time, through the same decode path that produced them,
before sampling continues. Restarting with different settings is refused.

Outputs: `prompt.txt` (the exact rendered prompt), `meta.json` (settings, input hash,
commit), `tokens.jsonl`, `reasoning.md` and `answer.md` (the two channels, rewritten
as tokens arrive), `progress.json`, `job.log`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Settings that must not change across a resume: they decide which tokens are sampled.
PINNED = ("messages_sha256", "prompt_tokens", "temperature", "top_p", "seed",
          "thinking_effort", "mode", "model_config_sha256")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


class Job:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.f_tokens = self.out / "tokens.jsonl"
        self.f_state = self.out / "state.npz"
        self.f_state_meta = self.out / "state.json"
        self.stop = False

    def log(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with (self.out / "job.log").open("a") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------ records
    def load_tokens(self) -> list[dict]:
        if not self.f_tokens.is_file():
            return []
        recs = []
        for line in self.f_tokens.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break                      # a torn last line from a hard stop
        # rewrite without any torn tail so appends stay line-aligned
        self.f_tokens.write_text("".join(json.dumps(r) + "\n" for r in recs))
        return recs

    def append_token(self, rec: dict) -> None:
        with self.f_tokens.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_channels(self, tok, ids: list[int]) -> None:
        from air_llm.models.kimi_k3.chat_format import ChannelParser
        parser = ChannelParser(tok, thinking=True)
        parts: dict[str, list[int]] = {"reasoning": [], "content": []}
        for i in ids:
            ch = parser.feed(i)
            if ch is not None:
                parts[ch].append(i)
        for ch, name in (("reasoning", "reasoning.md"), ("content", "answer.md")):
            tmp = self.out / (name + ".tmp")
            tmp.write_text(tok.decode(parts[ch]))
            tmp.replace(self.out / name)

    def write_progress(self, status: str, n: int, t0: float, rates: list[float]) -> None:
        recent = rates[-50:]
        rate = sum(recent) / len(recent) if recent else None
        left = self.a.max_tokens - n
        eta = rate * left if rate else None
        body = {
            "status": status, "tokens": n, "max_tokens": self.a.max_tokens,
            "session_elapsed": str(timedelta(seconds=int(time.time() - t0))),
            "recent_s_per_token": round(rate, 2) if rate else None,
            "eta_at_max_tokens": str(timedelta(seconds=int(eta))) if eta else None,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = self.out / "progress.json.tmp"
        tmp.write_text(json.dumps(body, indent=2))
        tmp.replace(self.out / "progress.json")

    def save_state(self, state, fed: int) -> None:
        tmp = self.out / "state.tmp.npz"
        state.save(tmp)
        tmp.replace(self.f_state)
        self.f_state_meta.write_text(json.dumps({"fed_tokens": fed}))

    # --------------------------------------------------------------------- run
    def run(self) -> int:
        import mlx.core as mx

        from air_llm.models.kimi_k3 import chat_format
        from air_llm.models.kimi_k3.config import K3Config
        from air_llm.models.kimi_k3.loader import TensorSource
        from air_llm.models.kimi_k3.model import KimiK3Model, _peak_memory, _reset_peak
        from air_llm.models.kimi_k3.recurrent_state import KimiK3State
        from air_llm.models.kimi_k3.sampling import choose, full_logits
        from air_llm.models.kimi_k3.tokenizer import load as load_tok

        a = self.a
        model_dir = Path(a.model)
        tok = load_tok(model_dir)
        messages = json.loads(Path(a.messages).read_text())
        segments = chat_format.render(messages, thinking_effort=a.thinking_effort)
        prompt = chat_format.encode(tok, segments)
        (self.out / "prompt.txt").write_text(chat_format.to_text(segments))

        meta = {
            "messages": str(a.messages),
            "messages_sha256": sha256(Path(a.messages)),
            "prompt_tokens": len(prompt),
            "temperature": a.temperature, "top_p": a.top_p, "seed": a.seed,
            "thinking_effort": a.thinking_effort, "mode": a.mode,
            "model_config_sha256": sha256(model_dir / "config.json"),
            "max_tokens": a.max_tokens, "resident_gb": a.resident_gb,
            "commit": git_commit(),
        }
        f_meta = self.out / "meta.json"
        old = json.loads(f_meta.read_text()) if f_meta.is_file() else None
        if old:
            diff = [k for k in PINNED if old.get(k) != meta[k]]
            if diff:
                self.log(f"refusing to resume: {diff} differ from {f_meta}")
                return 2

        # Check for a finished job before touching meta.json, so rerunning run.sh
        # over completed cases does not count as a resume.
        recs = self.load_tokens()
        ids = [r["id"] for r in recs]
        end_ids = {tok.special_tokens[n] for n in (chat_format.END_OF_MSG_TOKEN, "[EOS]")}
        if ids and ids[-1] in end_ids:
            self.log("already finished")
            self.write_channels(tok, ids)
            return 0
        if len(ids) >= a.max_tokens:
            self.log("already at --max-tokens")
            return 0

        if old:
            meta["started"] = old.get("started")
            meta["resumes"] = old.get("resumes", 0) + 1
            meta["commits"] = sorted(set(old.get("commits", [old.get("commit")]) + [meta["commit"]]))
        else:
            meta["started"] = datetime.now().isoformat(timespec="seconds")
            meta["resumes"] = 0
            meta["commits"] = [meta["commit"]]
        f_meta.write_text(json.dumps(meta, indent=2))

        cfg = K3Config.load(model_dir / "config.json")
        wdtype = mx.bfloat16 if a.mode == "runtime" else mx.float32
        src = TensorSource(manifest=Path(a.manifest), ckpt=model_dir,
                           cache=Path(a.cache) if a.cache else None,
                           resident_gb=a.resident_gb, weight_dtype=wdtype)
        model = KimiK3Model(cfg, src, max_memory_gb=a.max_memory_gb)

        state, fed = KimiK3State(), 0
        if ids and self.f_state.is_file() and self.f_state_meta.is_file():
            fed = json.loads(self.f_state_meta.read_text())["fed_tokens"]
            if len(prompt) <= fed < len(prompt) + len(ids):
                state = KimiK3State.load(self.f_state)
            else:
                fed = 0
        stream = prompt + ids
        self.log(f"prompt {len(prompt)} tokens, {len(ids)} generated so far, "
                 f"state covers {fed} of {len(stream)} tokens")

        def forward(chunk: list[int]):
            h = model.forward_token(model.embed(chunk), state)
            mx.eval(h)
            state.advance(len(chunk))
            return h

        t0, rates = time.time(), []
        self.write_progress("prefill" if fed < len(prompt) else "replaying", len(ids), t0, rates)
        # Bring the state up to date. The prompt goes in as one prefill, as in the
        # server; generated tokens are replayed one at a time, as they were produced.
        # A saved state never covers the last generated token, so `h` is always set here.
        if fed < len(prompt):
            t = time.time()
            h = forward(prompt)
            fed = len(prompt)
            self.log(f"prefill {len(prompt)} tokens in {time.time() - t:.0f}s")
        while fed < len(stream):
            if self.stop:
                return self.finish(tok, ids, "stopped", t0, rates)
            h = forward([stream[fed]])
            fed += 1
            self.log(f"replayed token {fed - len(prompt)}/{len(ids)}")

        while len(ids) < a.max_tokens:
            if self.stop:
                return self.finish(tok, ids, "stopped", t0, rates)
            t = time.time()
            _reset_peak()
            if h is None:
                h = forward([ids[-1]])
                fed += 1
            lg, _ = full_logits(model, h[:, -1, :])
            c = choose(lg, temperature=a.temperature, top_p=a.top_p,
                       seed=a.seed + len(ids))
            ids.append(c.token_id)
            h = None
            dt = time.time() - t
            rates.append(dt)
            self.append_token({"i": len(ids) - 1, "id": c.token_id,
                               "logprob": round(c.logprob, 5), "s": round(dt, 2),
                               "peak_gib": round(_peak_memory() / 2**30, 2)})
            if c.token_id in end_ids:
                return self.finish(tok, ids, "done", t0, rates)
            if len(ids) % a.save_every == 0:
                # The state covers everything but the token just sampled.
                self.save_state(state, fed)
            if len(ids) % 10 == 0:
                self.write_channels(tok, ids)
                self.write_progress("generating", len(ids), t0, rates)
                self.log(f"{len(ids)} tokens, {dt:.1f}s/token")
        return self.finish(tok, ids, "max_tokens", t0, rates)

    def finish(self, tok, ids, status, t0, rates) -> int:
        self.write_channels(tok, ids)
        self.write_progress(status, len(ids), t0, rates)
        self.log(f"{status}: {len(ids)} tokens")
        return 130 if status == "stopped" else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="manifest.jsonl.gz")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--resident-gb", type=float, default=8.0)
    ap.add_argument("--max-memory-gb", type=float, default=48.0)
    ap.add_argument("--mode", choices=["validation", "runtime"], default="runtime")
    ap.add_argument("--messages", required=True,
                    help="JSON list of chat messages ({role, content})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=12000)
    # Moonshot's published evaluation settings for single-step tasks.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking-effort", choices=["low", "high", "max"], default="max")
    ap.add_argument("--save-every", type=int, default=25,
                    help="save the state every N tokens; a resume replays at most N-1")
    a = ap.parse_args(argv)

    job = Job(a)

    def on_signal(signum, _frame):
        job.log(f"signal {signum}: stopping after the current token")
        job.stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    return job.run()


if __name__ == "__main__":
    raise SystemExit(main())
