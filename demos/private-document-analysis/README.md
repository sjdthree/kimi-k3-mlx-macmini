# Private-document analysis demo

Why run a 2.8T model at ~32 seconds per token when the cloud is fast and cheap? This demo
tests one answer: document-heavy analysis over material an organization may prefer not to
send to a cloud inference provider.

> **All three cases have now been run on the Mac mini.** The local answers are in
> [local-results/results.md](local-results/results.md). They took 3–4½ days of decode
> per case (Sep 29 – Oct 13, 2026), and every token was generated on the Mac by this
> repository's implementation.
>
> The cases were first run on the same model in the cloud (`moonshotai/kimi-k3` through
> OpenRouter, 2026-09-22), kept here as [cloud-reference/](cloud-reference/). The
> documents are *simulated* private material; nothing in them is actually confidential,
> so there was no reason not to send them to a cloud provider for a reference answer.

## The cases

Three synthetic "confidential" packages, each two pages, with a planted central
conclusion spread across several sections and some deliberate distractors. No real
company, patient or confidential data is involved.

| # | Case | Package | Prompt |
|---|---|---|---|
| 1 | Semiconductor root-cause investigation | [PDF](cases/01_semiconductor_confidential_excursion.pdf) | [prompt](prompts/semiconductor.txt) |
| 2 | Medical-AI adverse-event investigation | [PDF](cases/02_medical_ai_confidential_incident.pdf) | [prompt](prompts/medical_ai.txt) |
| 3 | R&D / trade-secret invention strategy | [PDF](cases/03_rd_tradesecret_invention_strategy.pdf) | [prompt](prompts/rd_tradesecret.txt) |

The complete raw answers are in [cloud-reference/results.md](cloud-reference/results.md)
and the run metadata in [cloud-reference/summary.json](cloud-reference/summary.json).
Both are kept byte-for-byte as received; [SHA256SUMS](SHA256SUMS) records their hashes.
[How the cases were built and judged](evaluation/methodology.md).

## Cloud reference run

| Case | Prompt tokens | Completion tokens | of which reasoning | Elapsed | Cost |
|---|---:|---:|---:|---:|---:|
| Semiconductor | 1,485 | 6,057 | 4,325 | 187.46 s | $0.0693 |
| Medical AI | 1,248 | 6,856 | 4,899 | 71.92 s | $0.1064 |
| R&D / trade secret | 1,461 | 9,821 | 6,567 | 98.59 s | $0.1513 |

Total cost for the three: **$0.327**. Maximum completion tokens: 20,000. All three
finished with `stop`.

The cloud is far faster and cheaper than this Mac. Use the cloud whenever the material
can go there. Local execution matters only when the corpus itself is the problem.

## Local run

| Case | Prompt tokens | Completion tokens | of which reasoning | Generation time | Median s/token | Finished |
|---|---:|---:|---:|---:|---:|---|
| Semiconductor | 1,278 | 7,623 | 5,763 | 72.6 h | 34.3 | 2026-10-10 |
| Medical AI | 1,039 | 8,225 | 5,943 | 79.9 h | 35.0 | 2026-10-13 |
| R&D / trade secret | 1,239 | 11,651 | 8,425 | 112.5 h | 34.9 | 2026-10-04 |

All three ended on their own, with K3's end-of-message token, well under the
20,000-token cap. Answer tokens (the rest of each completion, less 13 chat-format tokens)
came to 1,847 / 2,269 / 3,213. Peak MLX allocator memory was 22–24 GiB during prefill
and 12.4–13.2 GiB while decoding. The raw run files are in
[`runs/demo-local/`](../../runs/demo-local/): every token with its logprob and time
(`tokens.jsonl`), the reasoning channel, the answer, the rendered prompt, the job log and
the settings. [`local/collect_results.py`](local/collect_results.py) builds
[local-results/results.md](local-results/results.md) and
[summary.json](local-results/summary.json) from them.

Decode ran at 34–35 s/token, about 7% slower than the 32.4 s/token headline benchmark.
These answers ran to 9,000–13,000 tokens of context, while the benchmark was measured
on shorter runs.

**One interruption.** On 2026-10-05 at 11:13, about 32 hours into the semiconductor
case, the USB4 enclosure holding the checkpoint stopped responding. Its NVMe drive dropped
off the bus, the job exited with `OSError: [Errno 6] Device not configured`, and the
drive stayed offline until the Mac was rebooted. Per-token times had been flat until then,
with no stalls or slowdown. Before restarting, every checkpoint file was read back and
checked: all 98 weight and tokenizer files matched Hugging Face's published SHA-256 for
the pinned revision. The job resumed on 2026-10-08 from its last saved state (token
3,324), replayed the 25 tokens after it, and continued. Only the semiconductor case was
affected. Including that 70-hour outage, the three cases took 14 days of wall-clock time.

The local answers are longer than the cloud ones (7,623–11,651 completion tokens against
6,057–9,821), and the reasoning share is about the same (72–76% local, 67–72% cloud).

## Running the cases locally

The three cases run on the Mac mini in K3's own chat format with thinking on, as the
model was trained. The chat format is ported from Moonshot's
`encoding_k3.py` and checked against it token for token (`tests/test_chat_format.py`).

```sh
export K3_CKPT=/path/to/Kimi-K3 K3_MANIFEST=/path/to/manifest.jsonl.gz
sh demos/private-document-analysis/local/run.sh
```

| Setting | Local run | Cloud run |
|---|---|---|
| Input | PDF text from `pdftotext -layout`, then the question; committed in [`local/inputs/`](local/inputs/) by [`make_inputs.py`](local/make_inputs.py) | Not recorded |
| Prompt tokens | 1,278 / 1,039 / 1,239 | 1,485 / 1,248 / 1,461 |
| Sampling | Temperature 1.0, top-p 0.95 (Moonshot's published settings), seed 0 | Not recorded |
| Thinking effort | `max` (the upstream default) | Not recorded |
| Token cap | 20,000 | 20,000 |

The cloud prompts were 207–222 tokens longer in every case, so the cloud provider
probably wrapped the PDF text differently or added a system prompt. The local input
is therefore not identical to the cloud input, and the answers will differ word for
word even if the conclusions agree. The comparison that matters is whether the local
answers recover the same planted conclusions.

At ~35 s/token each case takes days. [`tools/run_chat_job.py`](../../tools/run_chat_job.py)
writes every token to disk as it goes, saves the model state every 25 tokens and
resumes exactly after an interruption. Rerunning `run.sh` resumes wherever it stopped.
Watch a run with `cat runs/demo-local/<case>/progress.json` and
`tail -f runs/demo-local/<case>/reasoning.md`.

## What the model found

The two runs are compared case by case below. "Planted" refers to the central
conclusions that were built into each package (see [the methodology](evaluation/methodology.md)).
The local answers were read against the same planted conclusions described here for the
cloud run. That reading was not checked against the private evaluator guide, and no
scores were assigned.

**R&D / trade secret.** The package planted four technical concepts that the simplified
existing patent claims do not cover: feed-forward cooling from the derivative of
workload power, online coolant-compressibility estimation, accumulator pre-charge
coupling, and per-unit microchannel geometry derived from post-assembly thermal maps.

- *Cloud:* identified all four (its concepts A–D). It argued that the combination of
  the first three is more defensible than any one of them, because the measured benefit
  disappears when pre-charge shifts. It also raised two points that the package does not
  state as conclusions. First, the supplier, which keeps many units' machining coordinates
  for 30 days, is a larger reverse-engineering channel than a customer. Second, the
  compressibility estimator could detect pre-charge drift and trigger recalibration. That
  is a *candidate* invention concept, not one of the planted answers. Like everything
  else here, it would need prior-art search and legal review, and the answer says so itself.
- *Local:* also identified all four (A–D) and made the same A+B+C combination argument,
  citing the >12% pre-charge sensitivity as evidence of synergy rather than mere
  aggregation. It flagged statistical aggregation of the supplier's coordinate streams as
  a residual leak path. Its own unplanted additions were different from the cloud's:
  - layering the compressibility estimator on HF-118's already-claimed pressure sensors
    as a "virtual sensor";
  - calibrating each unit's controller from its machining thermal map, which it labeled
    as an untested hypothesis;
  - treating the build-server / supplier data split itself as a protectable trade-secret
    system.

**Semiconductor.** The planted chain runs: chamber C2 maintenance, a mis-seated edge
gas-ring seal, radial nonuniformity, edge low-k thinning and CD bias, edge bridge
defects, and recovery after the seal was reseated.

- *Cloud:* reconstructed that chain and explained why a center-point post-maintenance
  qualification could pass while missing an edge-only problem. It downweighted the
  distractors (pattern density, photoresist lot, humidity, CMP, scanner overlay, chilled
  water). One of its proposed checks uses data that already exists: if the wafers already
  showing the full radial signature were etched before the 09:12 chilled-water event,
  that alternative is eliminated.
- *Local:* reconstructed the same chain, explained the same qualification escape, and
  set aside the same distractors. It used the density split (0.73 vs 0.69/cm²) to
  explicitly falsify the yield team's pattern-density suspicion. It proposed three
  wafer-free tests first:
  - azimuthal re-mapping of the existing W01/W02 data, since a mis-seated seal should
    leave an asymmetric signature;
  - seal seating-height metrology;
  - a leak / rate-of-rise test to separate an air leak from a flow redistribution.

  It also noted that defects track CD bias more closely than thickness.
- *Neither run* noticed that the package contradicts itself: its metrology table places
  W03 on chamber C1, while its tool history says W01–W06 ran on C2. Both used the
  table's labels. The local answer also refers to "all six C2 wafers" sharing a stable
  signature, but only W01 and W02 were measured.

**Medical AI.** The planted cause was a stale machine-readable clinical-policy
configuration, not a model or coding defect.

- *Cloud:* reconstructed the timing (revised protocol at 08:12, alert threshold crossed
  at 10:31, suppressed by the stale four-hour policy where the revised protocol implied
  two hours). It separated model, software, clinical-policy, change-control and human
  contributions, and it mapped contractual responsibility without turning it into a
  legal-liability conclusion.
- *Local:* reached the same verdicts in the same five categories and made the
  counterfactual explicit: a two-hour window would have closed at 10:12, so the 10:31
  alert would have been shown. It noted that the stale policy also governed the nurses'
  deferral instruction, so updating the software alone would not have been enough. It
  listed eight unresolved facts, including whether an earlier alert would have changed the
  outcome. It closed by stating that no liability conclusion follows from the package.

## What this does and does not show

It shows that Kimi-K3 recovered the planted central conclusions in three synthetic cases,
both in the cloud and running locally on a 64 GiB Mac mini through this repository's
implementation, and that it added useful hypotheses and diligence steps. The local run is
evidence that the implementation produces coherent, document-grounded long-form analysis
end to end: 7,600–11,700 tokens per answer, with thinking.

It does **not** show that Kimi-K3 is uniquely capable of this, that smaller models fail,
or that 2.8T parameters are needed. Each run is a single sample per case, and the reading
above is one reviewer's, not a blinded score. It is not a patentability, medical or legal
opinion. Local execution keeps the prompt and corpus away from a cloud inference
provider. It is not a general security guarantee.

The obvious next experiment is to run the same prompts through smaller open models and
score them the same way. See [the methodology](evaluation/methodology.md) for why these
three cases can no longer be used blind.
