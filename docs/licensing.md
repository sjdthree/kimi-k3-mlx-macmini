# Licensing and provenance review

The root MIT license applies to first-party contributions where indicated. It does
not replace upstream terms. Package metadata intentionally uses
`LicenseRef-Project-Mixed`; wheels include NOTICE and the separate license texts.
This is a provenance inventory and review checklist, **not a legal opinion**.

| Material | Origin | Treatment / outstanding review |
|---|---|---|
| First-party MLX/Metal implementation and tooling | Steve Derezinski | MIT, subject to identified copied/adapted portions below |
| `k3ref/` | Moonshot `modeling_kimi_linear.py` | Kimi K3 License; CPU reference adaptations. Upstream file identical in every public revision (see below) |
| `air_llm/models/kimi_k3/config.py` | Extracted from the reference adapter | Conservatively retain Moonshot attribution/Kimi terms |
| `air_llm/models/kimi_k3/tokenizer.py` PAT_STR | Moonshot `tokenization_kimi.py:54-63` | Verbatim pattern (confirmed by diff); Kimi license notice required |
| `air_llm/models/kimi_k3/chat_format.py` | Moonshot `encoding_k3.py` (revision `c5d1dd4c`, sha256 `b9cb7ae1…5333`) | Port of the text-chat subset, including the fixed thinking-effort message; Kimi K3 License. Segments and token ids checked against the shipped file by `tests/test_chat_format.py` |
| `k3ref/_vendor/kda_naive.py` | FLA `fla/ops/kda/naive.py` | MIT, original copyright and license retained; byte-identical to upstream (see below) |
| `archive/p0/p0_4_kda_layer.py` | Moonshot/FLA operations | Re-implements FLA gate/norm formulas with source anchors and loads the vendored FLA kernel; header points to NOTICE (reviewed 2026-09-21) |
| `p0/run_qwen3_mlx.py` (removed) | Adapted AirLLM's `Attention`/`TransformerBlock` (Apache License 2.0) | Removed before release; Qwen/GLM work moved to a separate repository. Still reachable in git history |
| Checkpoint/tokenizer data | Moonshot | Separate download, not redistributed here |

Sources: [Kimi model/license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE),
[FLA license](https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE),
[AirLLM license](https://github.com/lyogavin/airllm/blob/main/LICENSE).
Gavin Li's article, *Unbelievable! Run Kimi K3–2.8 Trillion Parameters — on a Single
4GB GPU* (AI Advances, 29 July 2026):
<https://aiadvances.org/unbelievable-run-kimi-k3-2-8-trillion-parameters-on-a-single-4gb-gpu-23590e7a16c2>.
It describes AirLLM as MIT-licensed; the AirLLM repository itself is Apache 2.0, and
this project follows the repository. Do not vendor a copy of the article.

## Upstream sources, verified 2026-09-21

- **flash-linear-attention.** `k3ref/_vendor/kda_naive.py` (git blob
  `51ff7076b26244366cf7ed2999ed13052da19cd5`) is byte-identical to
  `fla/ops/kda/naive.py` from commit
  [`14bef3d9`](https://github.com/fla-org/flash-linear-attention/blob/14bef3d9ec98ee359c9a9c7860795d91675dd091/fla/ops/kda/naive.py)
  (2026-04-17) through current main, including tags v0.5.0–v0.5.2. Line anchors cited
  in `k3ref/` and `archive/p0/` match FLA main from 2026-07-25, so the copy was most likely
  taken between 2026-07-25 and 2026-08-07 (around v0.5.2). The upstream LICENSE is
  identical to `licenses/LICENSE.flash-linear-attention`.
- **moonshotai/Kimi-K3.** Public and ungated, with 8 commits, the first
  [`c5d1dd4c`](https://huggingface.co/moonshotai/Kimi-K3/tree/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721)
  on 2026-07-27. `LICENSE`, `config.json`, `modeling_kimi_linear.py`,
  `tokenization_kimi.py` and every weight shard are unchanged across all of them, so
  the content this project used is the same whichever revision was downloaded. All 12
  source anchors in `k3ref/layers.py` line up with `modeling_kimi_linear.py`. The
  upstream LICENSE is identical to `licenses/LICENSE.kimi-k3`.
- **AirLLM.** Apache License 2.0 in every revision (`LICENSE` and `air_llm/LICENSE`),
  declared as Apache in `setup.py`, with no NOTICE file and no filled-in copyright
  line. The only file that adapted AirLLM code, `p0/run_qwen3_mlx.py`, was removed
  before release, as was `p0/validate_mlx.py`, which only imported AirLLM.
- **Date discrepancy.** Project notes and a former API alias `kimi-k3-2026-07-16` implied
  a download from 2026-07-16, but the first public Hugging Face commit is 2026-07-27, and
  the benchmark machine's copy of the checkpoint has file dates of 2026-07-27. The content
  is identical in either case. The alias was removed before release.

The checked-in Kimi license has conditional commercial provisions, including
model-as-a-service revenue and certain branding conditions. Have a human assess their
applicability to reference code, copied tokenizer material, redistributions and intended
uses. Do not advertise the entire distribution or model as unrestricted MIT.

## Required sign-off before public release

1. Resolve the exact upstream revisions used for vendored/adapted sources from original
   download records where possible. Do not substitute today's upstream HEAD as the
   historical origin. Compare content and record permanent URLs and hashes.
2. Review the tokenizer pattern, reference/configuration adapter, P0 KDA prototype,
   and AirLLM prototype for sufficient notices and applicable terms. Add any missing
   upstream text before redistribution. Unknown provenance remains a release gate.
3. Confirm README/package/wheel notices accurately describe the mixed distribution.
4. Confirm no full Gavin article or other restricted prose is reachable in git objects.
   The audit found no such article in reachable history; retain that history unless
   a concrete disclosure/licensing issue is subsequently identified.

Local source hashes below identify current repository content, not upstream origin:

- `k3ref/_vendor/kda_naive.py`: SHA-256 `60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016`
- `licenses/LICENSE.kimi-k3`: SHA-256 `20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887`
- `licenses/LICENSE.flash-linear-attention`: SHA-256 `41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7`
