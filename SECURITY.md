# Security

## Reporting a vulnerability

Please report security issues privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**. Do not open a public issue for
anything exploitable. Include the commit, the command or request that triggers the
problem, and what you expected instead.

This is a research project maintained by one person, so there is no response-time
guarantee. Reports are still read, and fixes are credited unless you ask otherwise.

## Scope and threat model

The code is written to run **on your own machine, for yourself**. Keep that in mind when
deciding what counts as a vulnerability:

- **The HTTP server is for local use.** It binds to `127.0.0.1` by default, requires an
  API key unless you set `AIR_LLM_ALLOW_ANONYMOUS=1`, and has no TLS. It has not been
  hardened for exposure to a network or the internet. Putting it on a public interface
  is outside the supported configuration.
- **Checkpoint files are treated as untrusted input.** The loader reads safetensors
  headers and byte ranges directly, rejects header sizes larger than the shard, checks
  that every tensor range lies inside its file, and loads saved state with
  `allow_pickle=False`. A crafted checkpoint that gets past these checks and causes
  memory corruption or code execution is in scope.
- **Ordinary inference runs no downloaded Python code.** The checkpoint ships
  `modeling_*.py` and `tokenization_kimi.py`; the runtime does not import them. The one
  exception is `tests/test_tokenizer.py`, which cross-checks the tokenizer against the
  shipped one via `transformers` with `trust_remote_code=True` — only run it against a
  checkpoint you trust.
- **Keys and content.** API keys are compared in constant time and never written to logs
  or metrics (a hashed key id is used instead). Prompt and completion text is not logged
  by the server, but `--token-log` records every generated token id and
  `tools/run_long_job.py` writes the prompt's output text, so treat those files as
  sensitive.

Out of scope: the upstream model's outputs, the upstream checkpoint itself, and
denial of service by a caller who already holds a valid key (one request can
legitimately occupy the machine for hours).
