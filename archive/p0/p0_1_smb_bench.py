"""P0-1: measure SMB read throughput against the Kimi-K3 checkpoint.

Two workloads that mirror how the streaming runtime will actually read:
  A. sequential bulk read  -> dense-weight staging (P0-5) and layer loads
  B. 5.5 MB random reads   -> per-expert MXFP4 tensor fetches (16 experts x 3 tensors x 92 layers)
"""
import json, os, random, struct, sys, time

CKPT = os.environ.get("K3_CKPT", "Kimi-K3")
random.seed(20260806)


def shards():
    return sorted(f for f in os.listdir(CKPT) if f.endswith(".safetensors"))


def seq_read(path, nbytes, offset=0, block=8 << 20):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.lseek(fd, offset, os.SEEK_SET)
        got = 0
        t0 = time.time()
        while got < nbytes:
            b = os.read(fd, min(block, nbytes - got))
            if not b:
                break
            got += len(b)
        return got, time.time() - t0
    finally:
        os.close(fd)


def expert_offsets(shard, n):
    """Real byte ranges of expert weight_packed tensors, from the safetensors header."""
    with open(os.path.join(CKPT, shard), "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    base = 8 + hlen
    out = []
    for k, v in hdr.items():
        if k == "__metadata__" or "weight_packed" not in k:
            continue
        s, e = v["data_offsets"]
        out.append((base + s, e - s))
    random.shuffle(out)
    return out[:n]


def main():
    sh = shards()
    print(f"shards: {len(sh)}")

    # A. sequential
    target = sh[10]
    p = os.path.join(CKPT, target)
    size = os.path.getsize(p)
    n = min(2 << 30, size)
    got, dt = seq_read(p, n, offset=size // 4)
    print(f"[A] sequential   {got/2**30:6.2f} GiB in {dt:7.2f}s = {got/dt/1e6:8.1f} MB/s  ({target})")

    # B. random expert-sized reads, spread across several shards
    picks = []
    for s in random.sample(sh[1:93], 6):
        for off, ln in expert_offsets(s, 12):
            picks.append((s, off, ln))
    random.shuffle(picks)
    total = 0
    t0 = time.time()
    lat = []
    for s, off, ln in picks:
        fd = os.open(os.path.join(CKPT, s), os.O_RDONLY)
        try:
            os.lseek(fd, off, os.SEEK_SET)
            t1 = time.time()
            got = 0
            while got < ln:
                b = os.read(fd, ln - got)
                if not b:
                    break
                got += len(b)
            lat.append(time.time() - t1)
            total += got
        finally:
            os.close(fd)
    dt = time.time() - t0
    lat.sort()
    print(f"[B] random {len(picks)} reads, avg {total/len(picks)/2**20:.2f} MiB each")
    print(f"    {total/2**30:6.2f} GiB in {dt:7.2f}s = {total/dt/1e6:8.1f} MB/s")
    print(f"    per-read latency  p50 {lat[len(lat)//2]*1e3:7.1f} ms   p90 {lat[int(len(lat)*.9)]*1e3:7.1f} ms")

    # projection
    rate = total / dt
    experts_bytes = 16 * 92 * 17_547_264
    print(f"\n[proj] experts/token = {experts_bytes/2**30:.1f} GiB -> {experts_bytes/rate:6.1f}s over SMB")
    print(f"[proj] dense/token   = 114 GiB -> {114*2**30/5e9:6.1f}s if local NVMe @5GB/s"
          f" | {114*2**30/rate:6.1f}s if left on SMB")


if __name__ == "__main__":
    main()
