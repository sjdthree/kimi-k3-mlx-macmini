> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../docs/evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# Software Specification: Kimi-K3 on Apple Silicon Using AirLLM and MLX

## 1. Objective

Implement full-text inference for the 1.56 TB Hugging Face Kimi-K3 checkpoint on Apple Silicon by extending AirLLM with:

- Kimi-K3 architecture support
- MLX-native model execution
- per-expert weight streaming from disk
- on-device MXFP4 decoding
- numerically correct KDA and MLA attention
- bounded unified-memory usage

Correctness and reliability take priority over throughput. Disk-bound generation measured in minutes per token is acceptable.

Kimi-K3 is a 2.8-trillion-parameter mixture-of-experts model with 896 routed experts, 16 selected experts per token, Kimi Delta Attention, MLA layers, Attention Residuals, and approximately 104 billion activated parameters per token. ([arXiv](https://arxiv.org/abs/2607.24653?utm_source=chatgpt.com))

## 2. Required Environment

### Hardware

- Apple Silicon Mac
- Sufficient internal or directly attached NVMe storage for the 1.56 TB checkpoint
- At least 64 GB unified memory recommended
- 128 GB unified memory preferred
- APFS filesystem
- At least 200 GB additional working space for indexes, manifests, test artifacts, and optional repacked expert files

### Software

- macOS
- Python 3.11 or 3.12
- Current stable MLX
- AirLLM development checkout
- Hugging Face Transformers
- Safetensors
- NumPy
- PyTest

MLX supports custom JIT-compiled Metal kernels through `mx.fast.metal_kernel`, which will be used where standard MLX operations are insufficient. ([ML Explore](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html?utm_source=chatgpt.com))

## 3. Repository Structure

```text
airllm/
    air_llm/
        backends/
            base.py
            mlx_backend.py

        models/
            kimi_k3/
                __init__.py
                config.py
                model.py
                attention_mla.py
                attention_kda.py
                router.py
                moe.py
                mxfp4.py
                recurrent_state.py
                weight_manifest.py

        streaming/
            tensor_loader.py
            expert_loader.py
            cache.py
            lifecycle.py

        kernels/
            metal/
                mxfp4_matvec.py
                kda_step.py
                normalization.py

    tools/
        build_kimi_k3_manifest.py
        verify_checkpoint.py
        export_cuda_reference.py
        compare_reference.py
        inspect_memory.py

    tests/
        unit/
        integration/
        parity/
        full_model/
```

## 4. Backend Dispatch

Replace platform-wide macOS routing with architecture-aware dispatch.

Required behavior:

```python
model = AutoModel.from_pretrained(
    checkpoint_path,
    backend="mlx",
    trust_remote_code=True,
)
```

Dispatch must inspect `config.model_type` and instantiate:

```python
AirLLMKimiK3Mlx
```

Required registry:

```python
MODEL_REGISTRY = {
    ("kimi_k3", "mlx"): AirLLMKimiK3Mlx,
}
```

The Kimi-K3 class must own all model semantics. AirLLM remains responsible for locating, loading, caching, and releasing tensors.

## 5. Checkpoint Manifest

Create a one-time manifest without modifying the original checkpoint.

The manifest must map every tensor to:

- safetensors shard
- byte offset where available
- tensor shape
- tensor dtype
- transformer layer
- tensor role
- expert number
- packed-weight group size
- scale tensor
- expected logical orientation

Example:

```json
{
  "model.layers.12.mlp.experts.417.gate_up_proj.weight_packed": {
    "file": "model-00871-of-01240.safetensors",
    "layer": 12,
    "expert": 417,
    "role": "gate_up",
    "dtype": "uint8",
    "shape": [3584, 4608],
    "group_size": 32,
    "scale_tensor": "model.layers.12.mlp.experts.417.gate_up_proj.weight_scale"
  }
}
```

The manifest builder must:

1. scan all checkpoint indexes and shards;
2. validate that every referenced file exists;
3. validate tensor shapes against `config.json`;
4. identify all routed and shared experts;
5. report missing, duplicated, or unexpected tensors;
6. complete without loading full tensors into memory.

## 6. Tensor Lifecycle

Define explicit lifecycle classes:

```python
class TensorHandle:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device_buffer: object | None

class LayerContext:
    layer_index: int
    resident_tensors: dict[str, TensorHandle]

class ExpertContext:
    layer_index: int
    expert_index: int
    tensors: dict[str, TensorHandle]
```

Every tensor must transition through:

```text
UNLOADED
→ HOST_MAPPED
→ MLX_RESIDENT
→ EXECUTED
→ RELEASED
```

Each transition must be logged when diagnostic mode is enabled.

The implementation must permit no retained references after release. Add explicit `mx.eval()` synchronization before releasing buffers whose execution is deferred.

## 7. Execution Schedule

For each generated token:

```text
Embedding
→ Transformer layer 0
→ Transformer layer 1
→ ...
→ Transformer layer N
→ Final normalization
→ LM head
→ Token selection
```

For each transformer layer:

```text
1. Load non-expert layer tensors.
2. Restore that layer’s recurrent or attention state.
3. Execute KDA or MLA.
4. Apply Attention Residual logic.
5. Compute router logits.
6. Select routed experts.
7. Stream and execute selected experts serially.
8. Execute shared experts.
9. Accumulate MoE output.
10. Complete residual and normalization operations.
11. Save updated recurrent state.
12. Release all layer and expert tensors.
```

Only recurrent state, attention cache, hidden state, tokenizer data, and small global tensors may remain resident between layers.

## 8. MLA Attention

Implement the 24 full-attention layers using MLX operations.

Required functionality:

- q-LoRA projection
- key/value latent projection
- rotary embeddings
- query/key normalization
- output gating
- causal masking
- prefill KV-cache construction
- single-token decode
- cache append and retrieval

Use a standard numerically stable attention path:

```python
scores = (q @ k.transpose(...)) * scale
scores = apply_causal_mask(scores)
probabilities = softmax(scores.astype(mx.float32), axis=-1)
output = probabilities.astype(v.dtype) @ v
```

Flash Attention is not required for acceptance.

Tests must cover:

- sequence length 1
- sequence length 2
- short prefill
- cached decode
- multiple attention heads
- grouped query layouts
- long enough input to exercise cache growth

## 9. KDA Attention

Implement Kimi Delta Attention in two stages.

### Stage 1: MLX Reference Path

Construct KDA entirely from standard MLX tensor operations.

It must expose diagnostic outputs for:

- projected query
- projected key
- projected value
- convolution output
- gates
- decay values
- state before update
- state after update
- attention output before projection
- final projected output

Support:

- sequence prefill
- recurrent one-token decode
- state initialization
- state serialization
- state restoration
- FP32 accumulation where required

### Stage 2: Metal Kernel

After Stage 1 passes all parity tests, implement an optional fused Metal kernel for recurrent single-token KDA.

Public interface:

```python
def kda_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    gates: mx.array,
    decay: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Return output and updated recurrent state."""
```

The standard MLX path must remain available as the correctness reference.

KDA is a recurrent linear-attention mechanism, so testing must compare state after every token, not only final logits. ([arXiv](https://arxiv.org/abs/2510.26692?utm_source=chatgpt.com))

## 10. MXFP4 Weight Support

Implement checkpoint-native MXFP4 decoding for:

- `weight_packed`
- `weight_scale`
- group size 32
- exact nibble ordering used by the checkpoint
- exact scale interpretation used by the checkpoint
- correct logical matrix orientation

### Reference Decoder

First implement a scalar NumPy decoder:

```python
def decode_mxfp4_reference(
    packed: np.ndarray,
    scales: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    ...
```

This decoder is the authoritative test oracle.

### MLX Decoder

Implement:

```python
def decode_mxfp4_mlx(
    packed: mx.array,
    scales: mx.array,
    output_shape: tuple[int, ...],
    output_dtype: mx.Dtype,
) -> mx.array:
    ...
```

### Fused Matvec

Implement a Metal kernel that performs:

```text
MXFP4 unpack
→ scale application
→ matrix-vector multiplication
→ FP32 accumulation
→ BF16 or FP16 output
```

Interface:

```python
def mxfp4_matvec(
    x: mx.array,
    packed_weight: mx.array,
    weight_scale: mx.array,
    output_features: int,
    input_features: int,
) -> mx.array:
    ...
```

The fused implementation must process one projection at a time and must not create a complete persistent BF16 copy of an expert.

## 11. Router and MoE Execution

Implement the checkpoint’s exact routing algorithm.

Required operations:

- router projection
- sigmoid score calculation
- correction-bias application
- top-16 expert selection
- routing-weight normalization
- shared-expert execution
- selected-expert weighted accumulation
- deterministic expert ordering

Expose:

```python
@dataclass
class RouterResult:
    expert_indices: mx.array
    expert_weights: mx.array
    raw_scores: mx.array
    corrected_scores: mx.array
```

For each selected expert:

```text
Load gate/up packed weights
→ execute gate/up projection
→ apply activation
→ load down packed weights
→ execute down projection
→ multiply by routing weight
→ accumulate in FP32
→ release expert buffers
```

The implementation must load selected experts serially by default.

## 12. Attention Residuals and Model State

Implement an explicit model-state object:

```python
@dataclass
class KimiK3State:
    token_position: int
    kda_states: dict[int, mx.array]
    mla_caches: dict[int, tuple[mx.array, mx.array]]
    attention_residuals: dict[int, mx.array]
```

Required features:

- initialize empty state;
- advance one token;
- save state to disk;
- restore state;
- verify restored generation matches uninterrupted generation;
- report state memory by component.

## 13. Numerical Modes

Support two modes.

### Validation Mode

- FP32 accumulation
- deterministic operations
- no fused experimental kernels
- full intermediate logging
- single-threaded expert loading
- fixed greedy decoding

### Runtime Mode

- BF16 or FP16 storage where validated
- fused MXFP4 matvec
- optional fused KDA step
- reduced logging
- bounded read-ahead

The same prompt and checkpoint must produce equivalent greedy tokens in both modes.

## 14. Logging and Diagnostics

Each token must produce a structured record:

```json
{
  "token_index": 3,
  "token_id": 1234,
  "elapsed_seconds": 212.8,
  "bytes_read": 93428172800,
  "peak_unified_memory": 82473902080,
  "layers_completed": 93,
  "experts_loaded": 1104,
  "cache_bytes": 318767104,
  "logit_checksum": "..."
}
```

Per-layer diagnostic mode must record:

- input hidden-state checksum
- attention output checksum
- router-selected experts
- router weights
- MoE output checksum
- output hidden-state checksum
- KDA state checksum
- current memory usage

No full hidden tensors should be written unless explicitly enabled.

## 15. Test Suite

### Unit Tests

| Component | Required test |
|---|---|
| Manifest | Every checkpoint tensor resolves to one valid source |
| MXFP4 nibble decode | Exhaustive test of all 16 FP4 codes |
| MXFP4 groups | Random and boundary-value groups |
| Scale decode | Exhaustive test of all 256 scale bytes |
| MXFP4 matvec | Compare against NumPy-expanded matrix |
| Router | Exact selected expert indices and weights |
| MLA | Compare against CPU reference |
| KDA step | Compare output and state against CPU reference |
| KDA sequence | Compare state after every token |
| Normalization | Compare against NumPy FP32 |
| RoPE | Compare known positions and frequencies |
| State save/restore | Bitwise-equivalent continuation where possible |
| Tensor lifecycle | No device references remain after release |

### Synthetic Integration Model

Create a miniature Kimi-K3 configuration retaining the same operations:

```text
hidden size: reduced
KDA layers: 2
MLA layers: 1
routed experts: 8
active experts: 2
shared experts: 1
vocabulary: reduced
MXFP4 group size: 32
```

Required tests:

1. eager full-memory execution;
2. AirLLM layer streaming;
3. AirLLM expert streaming;
4. standard MLX kernels;
5. fused Metal kernels;
6. saved-state continuation.

All paths must produce matching greedy output.

### Real-Checkpoint Component Tests

Using tensors from the full checkpoint:

- decode one real MXFP4 group;
- decode one complete projection row;
- execute one real expert projection;
- execute one real routed expert;
- execute one real shared expert;
- run one KDA layer;
- run one MLA layer;
- run one complete transformer layer;
- run two consecutive layers with state propagation.

### CUDA Reference Tests

Generate reference artifacts from a verified CUDA Kimi-K3 execution.

For a fixed prompt, capture:

- embeddings
- input and output of each transformer layer
- KDA recurrent state after each token
- MLA attention output
- router scores
- selected expert IDs
- routing weights
- MoE output
- final logits
- generated token IDs

Store reference tensors in a versioned manifest containing:

- checkpoint revision
- tokenizer revision
- source-code commit
- prompt
- random seed
- dtype settings
- tensor shape
- tensor checksum

## 16. Success Criteria

### Component Correctness

| Metric | Passing requirement |
|---|---:|
| MXFP4 decoded code values | Exact match before floating-point casting |
| Router expert selection | Exact top-16 match |
| Router weights | `atol ≤ 1e-5`, `rtol ≤ 1e-5` in FP32 validation |
| Individual FP32 operations | `atol ≤ 1e-5`, `rtol ≤ 1e-5` |
| BF16 layer output | Cosine similarity ≥ 0.999 |
| KDA recurrent state | Meets agreed per-step tolerance at every token |
| MLA output | Meets agreed tolerance before residual addition |
| State restore | Same subsequent greedy tokens as uninterrupted execution |

### Full-Model Correctness

A release candidate passes when it:

1. loads the complete 1.56 TB checkpoint;
2. validates every required tensor through the manifest;
3. completes full-model prefill for a fixed prompt;
4. generates at least 16 greedy tokens;
5. matches CUDA top-1 tokens for all 16 tokens;
6. matches CUDA router expert selection at every MoE layer;
7. maintains bounded numerical error across all layers;
8. produces coherent continuation on at least five additional prompts;
9. completes without process restart, memory exhaustion, or tensor corruption.

### Streaming Correctness

A release candidate must demonstrate that:

- only the current layer’s non-expert weights are resident;
- only one selected expert projection is expanded or executed at a time;
- released experts do not remain referenced;
- memory remains below a configured ceiling;
- disk reads correspond only to required layer tensors, selected experts, shared experts, and model state;
- repeated generation does not show sustained memory growth.

### Reliability Test

Run:

```text
5 prompts
× 16 generated tokens
× 3 independent process launches
```

Passing requirements:

- 15 of 15 runs complete;
- deterministic token output;
- no memory leak exceeding 2% between equivalent token positions;
- no corrupted reads;
- no NaN or infinity in hidden states, recurrent states, routing weights, or logits.

## 17. Implementation Milestones

| Milestone | Deliverable |
|---|---|
| M1 | Kimi-K3 dispatch and checkpoint manifest |
| M2 | NumPy reference operations and synthetic test model |
| M3 | MLX MLA implementation |
| M4 | MLX KDA reference implementation |
| M5 | MXFP4 decoder and reference matvec |
| M6 | Router, shared experts, and serial expert streaming |
| M7 | Full transformer-layer execution |
| M8 | Full-checkpoint one-token execution |
| M9 | CUDA parity for 16 greedy tokens |
| M10 | Fused Metal MXFP4 matvec and optional fused KDA |
| M11 | Reliability, memory, and state-restoration qualification |

## 18. Final Deliverables

The engineering team must provide:

- AirLLM Kimi-K3 MLX implementation
- checkpoint manifest builder
- checkpoint validation tool
- NumPy reference operations
- Metal MXFP4 matvec kernel
- KDA MLX implementation
- optional fused KDA Metal kernel
- CUDA-reference exporter
- parity comparison utility
- automated PyTest suite
- full-checkpoint run command
- memory and disk-I/O report
- reproducible environment lockfile
- implementation notes documenting checkpoint-specific tensor layouts
- final validation report against the success criteria above

## 19. Required Run Interface

```bash
python -m air_llm.run \
  --model /models/Kimi-K3 \
  --architecture kimi_k3 \
  --backend mlx \
  --stream-experts \
  --max-memory-gb 112 \
  --prompt-file prompts/test.txt \
  --max-new-tokens 16 \
  --temperature 0 \
  --validation-log results/run.jsonl
```

The implementation is complete when this command executes the full Hugging Face checkpoint on Apple Silicon and passes the full-model, parity, streaming, and reliability tests defined above.
