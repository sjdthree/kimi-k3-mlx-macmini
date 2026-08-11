Audit the following CUDA big-integer modular arithmetic library across three engineering
domains simultaneously. The code is a 256-bit Montgomery arithmetic backend intended for
elliptic-curve scalar multiplication on secp256k1.

```cpp
#include <cstdint>
#include <cuda_runtime.h>

#define LIMBS 4
#define WARP_SIZE 32
#define WINDOW_BITS 4

typedef struct { uint64_t v[LIMBS]; } bn256;

__constant__ uint64_t d_modulus[LIMBS];
__constant__ uint64_t d_n_prime;
__constant__ uint64_t d_r_squared[LIMBS];

// ---------------------------------------------------------------- limb primitives

__device__ __forceinline__ uint64_t addc(uint64_t a, uint64_t b, uint64_t* carry) {
    uint64_t s = a + b;
    *carry = (s < a) ? 1 : 0;
    return s;
}

__device__ __forceinline__ uint64_t subb(uint64_t a, uint64_t b, uint64_t* borrow) {
    uint64_t d = a - b;
    *borrow = (a < b) ? 1 : 0;
    return d;
}

__device__ __forceinline__ void mul_wide(uint64_t a, uint64_t b,
                                         uint64_t* lo, uint64_t* hi) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

// ---------------------------------------------------------------- comparison

__device__ int bn_cmp(const bn256* a, const bn256* b) {
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a->v[i] > b->v[i]) return 1;
        if (a->v[i] < b->v[i]) return -1;
    }
    return 0;
}

__device__ void bn_sub_mod(bn256* r, const bn256* a, const bn256* b) {
    uint64_t borrow = 0, b_tmp;
    for (int i = 0; i < LIMBS; i++) {
        r->v[i] = subb(a->v[i], b->v[i], &b_tmp);
        r->v[i] = subb(r->v[i], borrow, &borrow);
        borrow |= b_tmp;
    }
    if (borrow) {
        uint64_t carry = 0, c_tmp;
        for (int i = 0; i < LIMBS; i++) {
            r->v[i] = addc(r->v[i], d_modulus[i], &c_tmp);
            r->v[i] = addc(r->v[i], carry, &carry);
            carry |= c_tmp;
        }
    }
}

// ---------------------------------------------------------------- CIOS Montgomery

__device__ void mont_mul(bn256* r, const bn256* a, const bn256* b) {
    uint64_t t[LIMBS + 2] = {0};

    for (int i = 0; i < LIMBS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            uint64_t lo, hi;
            mul_wide(a->v[j], b->v[i], &lo, &hi);
            uint64_t s = t[j] + lo;
            if (s < t[j]) hi++;
            s += carry;
            if (s < carry) hi++;
            t[j] = s;
            carry = hi;
        }
        t[LIMBS] += carry;

        uint64_t m = t[0] * d_n_prime;
        uint64_t lo0, hi0;
        mul_wide(m, d_modulus[0], &lo0, &hi0);
        carry = hi0 + ((t[0] + lo0) < t[0] ? 1 : 0);

        for (int j = 1; j < LIMBS; j++) {
            uint64_t lo, hi;
            mul_wide(m, d_modulus[j], &lo, &hi);
            uint64_t s = t[j] + lo;
            if (s < t[j]) hi++;
            s += carry;
            if (s < carry) hi++;
            t[j - 1] = s;
            carry = hi;
        }
        t[LIMBS - 1] = t[LIMBS] + carry;
        t[LIMBS] = 0;
    }

    for (int i = 0; i < LIMBS; i++) r->v[i] = t[i];
    if (bn_cmp(r, (const bn256*)d_modulus) >= 0) {
        bn256 m_bn;
        for (int i = 0; i < LIMBS; i++) m_bn.v[i] = d_modulus[i];
        bn_sub_mod(r, r, &m_bn);
    }
}

// ---------------------------------------------------------------- exponentiation

__device__ void mont_pow(bn256* r, const bn256* base, const uint64_t* exp) {
    bn256 acc;
    for (int i = 0; i < LIMBS; i++) acc.v[i] = d_r_squared[i];

    bn256 table[1 << WINDOW_BITS];
    table[0] = acc;
    table[1] = *base;
    for (int i = 2; i < (1 << WINDOW_BITS); i++) {
        mont_mul(&table[i], &table[i - 1], base);
    }

    for (int bit = 255; bit >= 0; bit -= WINDOW_BITS) {
        for (int k = 0; k < WINDOW_BITS; k++) mont_mul(&acc, &acc, &acc);
        int idx = (exp[bit >> 6] >> (bit & 63)) & ((1 << WINDOW_BITS) - 1);
        mont_mul(&acc, &acc, &table[idx]);
    }
    *r = acc;
}

// ---------------------------------------------------------------- kernels

__global__ void ScalarMulKernel(const bn256* __restrict__ points,
                                const uint64_t* __restrict__ scalars,
                                bn256* __restrict__ out,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ bn256 s_cache[256];

    s_cache[threadIdx.x] = points[idx];
    __syncthreads();

    bn256 acc;
    mont_pow(&acc, &s_cache[threadIdx.x], &scalars[idx * LIMBS]);

    if (bn_cmp(&acc, (const bn256*)d_modulus) == 0) {
        return;
    }
    __syncthreads();

    out[idx] = acc;
}

__global__ void BatchInvertKernel(bn256* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        bn256 t = data[i];
        uint64_t e[LIMBS];
        for (int k = 0; k < LIMBS; k++) e[k] = d_modulus[k];
        e[0] -= 2;
        mont_pow(&data[i], &t, e);
    }
}
```

Provide a rigorous multi-domain audit covering:

* **CUDA architecture and hardware correctness.** Analyse warp divergence and the
  `__syncthreads()` placement in `ScalarMulKernel` — in particular whether the early
  `return` can deadlock the block. Assess shared-memory sizing against `blockDim.x`,
  the out-of-bounds risk from the missing `idx < n` guard, register pressure from the
  1 KB per-thread window table in `mont_pow`, and memory coalescing on the `bn256`
  struct-of-arrays access pattern.

* **Cryptographic and abstract-algebra validity.** Evaluate the CIOS Montgomery
  reduction in `mont_mul` for correctness: the carry propagation through `t[LIMBS]`,
  the shifted-store `t[j-1]`, and whether the final conditional subtraction is
  sufficient to guarantee a fully reduced result. Assess whether `bn_sub_mod`'s
  borrow handling (`borrow |= b_tmp`) is correct. Evaluate the windowed exponentiation
  in `mont_pow`: the initial accumulator, the window-index extraction when a window
  straddles a 64-bit limb boundary, and whether 256 is divisible by `WINDOW_BITS`.
  Determine whether `BatchInvertKernel` computes a correct modular inverse via
  Fermat's little theorem, including the `e[0] -= 2` borrow case.

* **Side-channel and formal safety.** Identify every data-dependent branch that leaks
  timing information about secret scalars, and state which are exploitable. Provide
  corrected constant-time replacements for `bn_cmp` and the conditional subtraction.
  State the loop invariants a formal proof of `mont_mul` would need.

Ground every finding in the specific line or expression it concerns, and state clearly
which defects are exploitable versus merely incorrect.
