"""Run a multi-hour generation as a set-and-forget job.

A 2,500-token response at ~34 s/token is roughly 24 hours. Over that span the things
that actually go wrong are mundane: the terminal closes, the machine sleeps, the server
restarts, or something dies at hour 20. So this script is built around three properties:

1. **Every token hits disk immediately.** `output.txt` is appended and flushed per token.
   If the job dies at hour 23 you keep 23 hours of work.
2. **It can resume.** The server cannot resume a generation mid-stream, but the job can:
   re-send the original prompt with the text generated so far appended, and ask only for
   the remaining tokens. That costs one prefill (~10-15 min, growing as output
   accumulates) instead of losing everything.
3. **It is inspectable while running.** `progress.json` is rewritten every token with
   rate, ETA, and counts, so `cat progress.json` answers "how is it going" without
   attaching to the process.

Ctrl-C (or SIGTERM) closes the stream, which cancels the generation server-side within
~1 s rather than leaving it burning disk.

    python tools/run_long_job.py --prompt-file prompts/04_polymath_multi_expert.md \
        --out runs/polymath --max-tokens 2500

Check on it from anywhere:

    cat runs/polymath/progress.json
    tail -f runs/polymath/output.txt
    tail -f runs/polymath/job.log
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


class Job:
    def __init__(self, a):
        self.a = a
        self.out_dir = Path(a.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.f_text = self.out_dir / "output.txt"
        self.f_prog = self.out_dir / "progress.json"
        self.f_log = self.out_dir / "job.log"
        self.f_meta = self.out_dir / "meta.json"
        self.stop = False
        self.stream = None
        self.t0 = time.time()
        self.ntok = 0
        self.ttft = None
        self.resumes = 0

    # ------------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with self.f_log.open("a") as fh:
            fh.write(line + "\n")

    def write_progress(self, state: str, extra: dict | None = None) -> None:
        el = time.time() - self.t0
        rate = self.ntok / el if el > 0 and self.ntok else 0.0
        remaining = max(self.a.max_tokens - self.total_tokens(), 0)
        eta_s = remaining / rate if rate > 0 else None
        p = {
            "state": state,
            "prompt_file": str(self.a.prompt_file),
            "tokens_generated_total": self.total_tokens(),
            "tokens_this_attempt": self.ntok,
            "target_tokens": self.a.max_tokens,
            "percent": round(100 * self.total_tokens() / max(self.a.max_tokens, 1), 2),
            "elapsed_s": round(el, 1),
            "elapsed_human": str(timedelta(seconds=int(el))),
            "seconds_per_token": round(1 / rate, 1) if rate > 0 else None,
            "ttft_s": round(self.ttft, 1) if self.ttft else None,
            "eta_human": str(timedelta(seconds=int(eta_s))) if eta_s else None,
            "eta_finish": (datetime.now() + timedelta(seconds=eta_s)).isoformat(
                timespec="seconds") if eta_s else None,
            "resumes": self.resumes,
            "started": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "output_bytes": self.f_text.stat().st_size if self.f_text.exists() else 0,
        }
        if extra:
            p.update(extra)
        tmp = self.f_prog.with_suffix(".tmp")
        tmp.write_text(json.dumps(p, indent=2))
        tmp.replace(self.f_prog)          # atomic: a reader never sees a half file

    # -------------------------------------------------------------------- state
    def prior_text(self) -> str:
        return self.f_text.read_text() if self.f_text.exists() else ""

    def total_tokens(self) -> int:
        """Tokens on disk from all attempts. Approximated across resumes by count."""
        return self.baseline_tokens + self.ntok

    # ---------------------------------------------------------------------- run
    def run(self) -> int:
        from openai import OpenAI

        prompt = Path(self.a.prompt_file).read_text()
        client = OpenAI(base_url=self.a.base_url, api_key=self.a.api_key, timeout=None)

        prior = self.prior_text()
        if prior and not self.a.resume:
            self.log(f"ERROR: {self.f_text} already has {len(prior)} bytes. "
                     f"Pass --resume to continue it, or --out to a fresh directory.")
            return 2
        self.baseline_tokens = self.a.resume_token_count if prior else 0
        if prior:
            self.resumes = 1
            self.log(f"resuming: {len(prior)} bytes / ~{self.baseline_tokens} tokens "
                     f"already on disk")

        self.f_meta.write_text(json.dumps({
            "prompt_file": str(self.a.prompt_file),
            "model": self.a.model, "base_url": self.a.base_url,
            "max_tokens": self.a.max_tokens,
            "started": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))

        for attempt in range(1, self.a.retries + 2):
            if self.stop:
                break
            remaining = self.a.max_tokens - self.total_tokens()
            if remaining <= 0:
                self.log("target reached")
                break

            messages = [{"role": "user", "content": prompt}]
            prior = self.prior_text()
            if prior:
                # Resume by replaying: the server cannot continue a finished stream, but
                # feeding back the partial answer makes the model continue from it.
                messages.append({"role": "assistant", "content": prior})

            self.log(f"attempt {attempt}: requesting {remaining} more tokens "
                     f"(prefill will take ~10-20 min for a long prompt)")
            self.write_progress("prefill")

            try:
                if self.run_stream(client, messages, remaining):
                    self.log("generation finished normally")
                    break
            except KeyboardInterrupt:
                self.stop = True
                break
            except Exception as e:
                self.log(f"attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt > self.a.retries:
                    self.write_progress("failed", {"error": type(e).__name__})
                    return 1
                self.log(f"retrying in {self.a.retry_delay}s ...")
                for _ in range(self.a.retry_delay):
                    if self.stop:
                        break
                    time.sleep(1)
                self.resumes += 1

        state = "cancelled" if self.stop else "done"
        self.write_progress(state)
        self.log(f"{state}: {self.total_tokens()} tokens, "
                 f"{timedelta(seconds=int(time.time()-self.t0))} elapsed, "
                 f"output in {self.f_text}")
        return 0

    def run_stream(self, client, messages, remaining) -> bool:
        """Returns True if the model finished on its own."""
        t_req = time.time()
        self.stream = client.chat.completions.create(
            model=self.a.model, messages=messages, max_tokens=remaining,
            temperature=self.a.temperature, stream=True,
            stream_options={"include_usage": True})

        finished = False
        last_flush = 0.0
        with self.f_text.open("a", buffering=1) as fh:
            for chunk in self.stream:
                if self.stop:
                    self.log("stop requested -- closing stream (cancels server-side)")
                    self.stream.close()
                    return False
                if chunk.usage:
                    self.log(f"usage: prompt={chunk.usage.prompt_tokens} "
                             f"completion={chunk.usage.completion_tokens}")
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                if ch.delta.content:
                    if self.ttft is None:
                        self.ttft = time.time() - t_req
                        self.log(f"first token after {self.ttft:.1f}s "
                                 f"({self.ttft/60:.1f} min of prefill)")
                    self.ntok += 1
                    fh.write(ch.delta.content)
                    fh.flush()
                    os.fsync(fh.fileno())      # survive a hard power loss, not just a crash
                    now = time.time()
                    if now - last_flush >= self.a.progress_every:
                        self.write_progress("generating")
                        last_flush = now
                    if self.ntok % self.a.log_every == 0:
                        el = time.time() - self.t0
                        self.log(f"{self.total_tokens()}/{self.a.max_tokens} tokens "
                                 f"({100*self.total_tokens()/self.a.max_tokens:.1f}%) "
                                 f"{el/max(self.ntok,1):.1f}s/tok")
                if ch.finish_reason:
                    self.log(f"finish_reason={ch.finish_reason}")
                    finished = ch.finish_reason in ("stop", "length")
        return finished


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_long_job",
        description="Run a multi-hour generation with per-token durability and resume.")
    ap.add_argument("--prompt-file", required=True, type=Path)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--model", default="chat")
    ap.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-k3"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing output.txt in --out")
    ap.add_argument("--resume-token-count", type=int, default=0,
                    help="tokens already generated (for accurate ETA after a resume)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-delay", type=int, default=60)
    ap.add_argument("--log-every", type=int, default=10, help="log every N tokens")
    ap.add_argument("--progress-every", type=float, default=30.0,
                    help="rewrite progress.json at most every N seconds")
    a = ap.parse_args(argv)

    job = Job(a)

    def handle(signum, _frame):
        job.log(f"signal {signal.Signals(signum).name} -- finishing current token, "
                f"then cancelling")
        job.stop = True

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    job.log("=" * 70)
    job.log(f"job start  prompt={a.prompt_file}  target={a.max_tokens} tokens")
    job.log(f"  at ~34 s/token that is ~{a.max_tokens*34/3600:.1f} h of decoding, "
            f"plus prefill")
    job.log(f"  output dir: {Path(a.out).resolve()}")
    job.log("=" * 70)
    return job.run()


if __name__ == "__main__":
    raise SystemExit(main())
