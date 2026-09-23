# Private-document analysis demo

Why run a 2.8T model at ~32 seconds per token when the cloud is fast and cheap? This demo
tests one answer: document-heavy analysis over material an organization may prefer not to
send to a cloud inference provider.

> **These answers came from the same model running in the cloud, not from the Mac.**
> The documents are *simulated* private material: nothing in them is actually
> confidential, so there was no reason not to send them to a cloud provider. The
> reference run used `moonshotai/kimi-k3` through OpenRouter on 2026-09-22. It shows
> what the model does on these cases. It does not show local output.
>
> This demo is a side test. Every other result in the repository (benchmarks, parity,
> memory, sample output) comes from the Mac mini. The cases have **not yet been run on
> the local implementation**. At ~32.4 s/token, the 6,057–9,821 completion tokens of
> these answers would take roughly 2½–4 days of decode per case.

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

## What the model found

**R&D / trade secret.** The package planted four technical concepts that the simplified
existing patent claims do not cover: feed-forward cooling from the derivative of
workload power, online coolant-compressibility estimation, accumulator pre-charge
coupling, and per-unit microchannel geometry derived from post-assembly thermal maps.
The answer identified all four (its concepts A–D). It also argued that the combination
of the first three is more defensible than any one of them, because the measured
benefit disappears when pre-charge shifts. It raised two points that the package does
not state as conclusions:

- the supplier, which keeps many units' machining coordinates for 30 days, is a larger
  reverse-engineering channel than a customer;
- the compressibility estimator could detect pre-charge drift and trigger recalibration.
  This is a *candidate* invention concept. It was not among the planted answers, and
  like everything else here it would need prior-art search and legal review. The answer
  says so itself.

**Semiconductor.** The answer reconstructed the planted chain: chamber C2 maintenance,
a mis-seated edge gas-ring seal, radial nonuniformity, edge low-k thinning and CD bias,
edge bridge defects, and recovery after the seal was reseated. It explained why a
center-point post-maintenance qualification could pass while missing an edge-only
problem, and it downweighted the distractors (pattern density, photoresist lot, humidity,
CMP, scanner overlay, chilled water). One of its proposed checks uses data that already
exists: if the wafers already showing the full radial signature were etched before the
09:12 chilled-water event, that alternative is eliminated.

**Medical AI.** The planted cause was a stale machine-readable clinical-policy
configuration, not a model or coding defect. The answer reconstructed the timing
(revised protocol at 08:12, alert threshold crossed at 10:31, suppressed by the stale
four-hour policy where the revised protocol implied two hours). It separated model,
software, clinical-policy, change-control and human contributions, and it mapped
contractual responsibility without turning it into a legal-liability conclusion.

## What this does and does not show

It shows that Kimi-K3 recovered the planted central conclusions in three synthetic cases
and added useful hypotheses and diligence steps.

It does **not** show that Kimi-K3 is uniquely capable of this, that smaller models fail,
or that 2.8T parameters are needed. It is not a patentability, medical or legal opinion.
Local execution keeps the prompt and corpus away from a cloud inference provider. It is
not a general security guarantee.

The obvious next experiment is to run the same prompts through smaller open models and
score them the same way. See [the methodology](evaluation/methodology.md) for why these
three cases can no longer be used blind.
