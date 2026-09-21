# Licensing and provenance review

The root MIT license applies to first-party contributions where indicated. It does
not replace upstream terms. Package metadata intentionally uses
`LicenseRef-Project-Mixed`; wheels include NOTICE and the separate license texts.
This is a provenance inventory and review checklist, **not a legal opinion**.

| Material | Origin | Treatment / outstanding review |
|---|---|---|
| First-party MLX/Metal implementation and tooling | Steve Derezinski | MIT, subject to identified copied/adapted portions below |
| `k3ref/` | Moonshot `modeling_kimi_linear.py` | Kimi K3 License; CPU reference adaptations; exact originating snapshot unrecorded |
| `air_llm/models/kimi_k3/config.py` | Extracted from the reference adapter | Conservatively retain Moonshot attribution/Kimi terms |
| `air_llm/models/kimi_k3/tokenizer.py` PAT_STR | Moonshot `tokenization_kimi.py:54-63` | Verbatim pattern; Kimi license notice required |
| `third_party/kda_naive.py` | FLA `fla/ops/kda/naive.py` | MIT, original copyright and license retained; upstream revision unrecorded |
| `p0/p0_4_kda_layer.py` | Moonshot/FLA operations | Human review of adaptation scope; preserve source anchors and both notices |
| `p0/run_qwen3_mlx.py` | AirLLM classes/streaming prototype | Human review of copied versus imported portions; preserve AirLLM credit |
| Checkpoint/tokenizer data | Moonshot | Separate download, not redistributed here |

Sources: [Kimi model/license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE),
[FLA license](https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE),
[AirLLM license](https://github.com/lyogavin/airllm/blob/main/LICENSE).
Gavin Li's article is titled *Unbelievable! Run Kimi K3–2.8 Trillion Parameters —
on a Single 4GB GPU*. Link to the canonical author publication once confirmed;
do not vendor a full copy of the article.

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

- `third_party/kda_naive.py`: SHA-256 `60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016`
- `third_party/LICENSE.kimi-k3`: SHA-256 `20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887`
- `third_party/LICENSE.flash-linear-attention`: SHA-256 `41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7`
