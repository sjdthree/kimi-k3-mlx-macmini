# Feasibility spikes (unsupported)

These scripts answered the project's first go/no-go questions before any MLX code was
written. They are kept as a record. They are not the supported entry point and may need
extra dependencies. Results are in [P0_RESULTS.md](../P0_RESULTS.md).

| Script | Question it answered |
|---|---|
| `p0_1_smb_bench.py`, `p0_1b_parallel.py` | Can the checkpoint stream from a NAS over SMB? No: 36–48 MB/s random reads, and concurrency doesn't help. This result led to using a local USB4 NVMe drive. |
| `p0_2_alog.py` | Is KDA's `A_log [128]` per head or per channel? Per head, zero-padded from 96. |
| `p0_3_mxfp4.py` | Does a NumPy MXFP4 decoder match `compressed-tensors` bit for bit? Yes. |
| `p0_4_kda_layer.py` | Does one real KDA layer run on the CPU, finite and deterministic? Yes. |
| `k3io.py` | Minimal safetensors reader shared by the scripts above. |

For the supported path, see [INSTALL.md](../../INSTALL.md).
