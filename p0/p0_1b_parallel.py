"""P0-1b: does concurrency rescue the SMB random-read path?

Serial 5.25 MiB reads measured 36 MB/s at p50 141 ms -> latency-bound, not bandwidth-bound.
If so, N concurrent readers should scale until the ~102 MB/s sequential ceiling.
"""
import json, os, random, struct, time
from concurrent.futures import ThreadPoolExecutor

CKPT = "/Volumes/LLMModels/Kimi-K3"
random.seed(7)


def expert_ranges(shard, n):
    with open(os.path.join(CKPT, shard), "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    base = 8 + hlen
    out = [(base + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
           for k, v in hdr.items() if k != "__metadata__" and "weight_packed" in k]
    random.shuffle(out)
    return [(shard, o, l) for o, l in out[:n]]


def read_one(job):
    shard, off, ln = job
    fd = os.open(os.path.join(CKPT, shard), os.O_RDONLY)
    try:
        os.lseek(fd, off, os.SEEK_SET)
        got = 0
        while got < ln:
            b = os.read(fd, ln - got)
            if not b:
                break
            got += len(b)
        return got
    finally:
        os.close(fd)


def main():
    sh = sorted(f for f in os.listdir(CKPT) if f.endswith(".safetensors"))
    for workers in (4, 8, 16, 32):
        jobs = []
        for s in random.sample(sh[1:93], 8):
            jobs += expert_ranges(s, 12)
        random.shuffle(jobs)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            total = sum(ex.map(read_one, jobs))
        dt = time.time() - t0
        rate = total / dt
        print(f"workers={workers:3d}  {len(jobs):3d} reads  {total/2**30:5.2f} GiB in {dt:6.2f}s"
              f" = {rate/1e6:7.1f} MB/s   -> experts/token {24.1*2**30/rate:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
