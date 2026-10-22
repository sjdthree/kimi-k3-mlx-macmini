# Methodology

## Case construction

Each case is a two-page synthetic package (generated with ReportLab) about a fictional
organization. The central conclusion is never stated in one place. It has to be assembled
from several sections: metrology plus tool history plus a cross-lot comparison in case 1;
a policy excerpt plus software notes plus a timeline in case 2; notebook excerpts plus
manufacturing notes plus a firmware table in case 3. Each package also contains plausible
distractors.

A separate evaluator guide records the planted answer for each case and a scoring rubric.
It was never given to the model.

## The run

| | |
|---|---|
| Model | `moonshotai/kimi-k3` via OpenRouter (cloud, not the local implementation) |
| Date | 2026-09-22T19:49:28 |
| Max completion tokens | 20,000 |
| Prompts | [`prompts/`](../prompts/), taken verbatim from the "Question" sections of [the raw results](../cloud-reference/results.md) |

Each PDF ends with a short "Challenge prompt". That is a condensed version of the prompt.
The text in `prompts/` is what was actually sent.

Not recorded with this run: how the PDF contents were passed to the model (extracted text
or file attachment), the sampling parameters, any system prompt, and the OpenRouter
provider that served the request. Treat the run as a single sample, not a controlled
measurement.

## The local run

| | |
|---|---|
| Model | Full Kimi-K3 checkpoint (revision `c5d1dd4c`), this repository's MLX implementation, M4 Pro Mac mini with 64 GiB |
| Dates | 2026-09-29 to 2026-10-13 |
| Inputs | [`local/inputs/`](../local/inputs/): `pdftotext -layout` text plus the question from `prompts/`, as one user message with no system prompt |
| Settings | K3 chat format, thinking effort `max`, temperature 1.0, top-p 0.95, seed 0, 20,000-token cap |
| Results | [`local-results/`](../local-results/); raw files in [`runs/demo-local/`](../../../runs/demo-local/) |

Everything the cloud run left unrecorded is fixed and committed here. It is still one
sample per case. The local answers were read against the planted conclusions as the
demo README describes them, not against the evaluator guide, and were not scored.

## Judging

The answers were read against the evaluator guide on five dimensions, each scored 0–2:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Ground-truth discovery | Misses the planted mechanism or concepts | Partial | Finds the central planted conclusions |
| Cross-document synthesis | Treats sections independently | Some linkage | Builds explicit evidence chains |
| Distractor resistance | Chases distractors | Mixed | Correctly downweights distractors |
| Uncertainty discipline | Invents facts or overclaims | Some caveats | Separates evidence, inference and missing facts |
| Actionability | Generic recommendations | Some useful steps | Minimal discriminating tests, controls or diligence |

The claims in [the demo README](../README.md) come from one reviewer's reading of the
answers. No per-dimension scores were recorded, and there was no second, blinded grader.

## Blinding

The evaluator guide is not published. That no longer keeps the cases blind: the raw
answers and the summary in the demo README describe the planted conclusions in detail.
Any model trained on public data after this release may have seen them. A future
comparison against smaller models should use **new cases built the same way**, with
their answer key kept private until the comparison is finished. These three cases stay
useful as worked examples and for runs where contamination does not matter, such as
the local-versus-cloud comparison of the same model.
