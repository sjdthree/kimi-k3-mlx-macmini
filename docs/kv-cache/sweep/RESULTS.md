> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../../evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# `--resident-gb` sweep — 2026-08-10

Prompt 04 (291 tokens), 40 generated tokens per config, decode context 296–334. Only
`--resident-gb` moves; `--max-memory-gb 58` and everything else held constant. The first
decode token after prefill is dropped from every mean — it carries warming cost.

| resident | mean s/tok | min | max | hit rate | peak | throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 40 GiB | 69.00 | 61.3 | 77.3 | 0.426 | 46.2 GiB | 1.27 GiB/s |
| 32 GiB | 42.40 | 39.5 | 49.6 | 0.345 | 39.0 GiB | 2.25 GiB/s |
| 24 GiB | 33.16 | 32.8 | 34.1 | 0.237 | 31.0 GiB | 3.12 GiB/s |
| 16 GiB | 32.79 | 32.6 | 33.7 | 0.191 | 23.0 GiB | 3.40 GiB/s |
| **8 GiB** | **32.44** | 32.0 | 35.5 | 0.098 | **15.0 GiB** | **3.70 GiB/s** |

## The result is inverted, monotonically, across a 5× range

More pinned memory buys a **higher cache hit rate** and **worse throughput**, at every
step without exception. 8 GiB is **2.13× faster** than 40 GiB while hitting a quarter as
often.

`--resident-gb` is fill-and-pin, so it takes unified memory *permanently* away from the OS
page cache. The dense working set is 105.7 GiB on the internal SSD and the machine has
64 GiB, so the page cache is doing the heavy lifting — pinning 40 GiB starves it. The app
cache also serves only what it loaded **first**, whereas the page cache holds what is
actually being **used**. A miss served from page cache beats a hit served from swap.

## Per-token cost is now a one-line model

Every decode step reads a fixed 95.6 GiB (`bytes_read` is identical on every step — see
`../README.md`). So:

    s/token  ≈  95.6 GiB ÷ achievable read bandwidth  +  ~2 s compute

    8 GiB:  95.6 / 3.70 = 25.8 s  + overhead  ≈ 32.4   ✓ measured 32.44
    40 GiB: 95.6 / 1.27 = 75.3 s               ≈ 69–75 ✓ measured 69.00

Pinning memory only ever *reduces* the bandwidth term. There is nothing left to win this
way: the floor is ~32 s/token, and the next real gain must come from **reading fewer
bytes**, not from caching them better.

0 GiB was not tested. The trend suggests it would be marginally better again, but returns
had collapsed to ~1% per halving by 16→8.

## Two corrections to the record

**1. This reverses `SESSION.md` open thread 5**, which read *"More pinned memory is the
clearest remaining speedup — 32 GiB covers ~30% of the 105.7 GiB dense working set;
~40 GiB should reach ~31 s/token."* Measured, 40 GiB reaches 69 s/token — worse than every
smaller setting.

It also contradicts `tests/test_resident_cache.py`, which predicts 32 → 29.6 s/token and
40 → 27.2 s/token. That model computes bytes-streamed analytically and has **no term for
physical memory limits or page-cache displacement**, so it is optimistic in exactly the
regime that decides the answer. The test still passes because it is self-consistent; it is
the model that is wrong, not the assertion.

**2. An earlier retraction was over-cautious.** A resident-40 measurement of 67.0 s/token
was taken while a stale server was still running, and it was withdrawn as contaminated.
The clean re-run gives **69.0**. The contamination was real — a stale server *was*
competing — but it was not what produced the slow number. The original figure was
substantially correct.

## Process hygiene

`sweep.sh` exists because the contamination above was not caught for hours.
`pkill -f air_llm.serve` sends SIGTERM; uvicorn handles it by closing the listening
socket, and the process then hangs in an uninterruptible read off the USB4 volume and
never exits. **The port frees while the process keeps consuming disk bandwidth**, so "the
new server bound 8080 and became ready" is not evidence the old one died.

Every run therefore escalates to SIGKILL, asserts zero servers and zero jobs before
starting, asserts exactly one server before and after the measurement, and aborts rather
than measure a dirty machine.
