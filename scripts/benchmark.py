"""Efficiency study: how do the mixers scale with sequence length on *your* GPU?

Three measurements, each written to ``results/benchmark/*.csv``:

1. ``layer``  -- single-layer forward and forward+backward latency and peak memory vs sequence
                 length (the training-time view: O(T^2) attention vs O(T) scans).
2. ``scan``   -- the SSM scan alone: recurrent PyTorch loop vs chunked PyTorch vs the Triton
                 kernel, plus achieved memory bandwidth for the Triton kernel (roofline input).
3. ``decode`` -- per-token generation latency and cache/state size as the context grows
                 (the inference-time view: growing KV cache vs constant recurrent state).

    python scripts/benchmark.py --which layer scan decode
    python scripts/benchmark.py --which layer --lengths 512 1024 2048 4096 8192 16384
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fadingmem.layers.attention import CausalSelfAttention  # noqa: E402
from fadingmem.layers.deltanet import GatedDeltaNet  # noqa: E402
from fadingmem.layers.ssm import SelectiveSSM, ssd_scan_chunked, ssd_scan_recurrent  # noqa: E402


def timeit(fn, device, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def measure(fn, device, **kw):
    """Returns (ms, peak MB) or (None, None) on out-of-memory."""
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        ms = timeit(fn, device, **kw)
        mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        return ms, mem
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def mixers(d_model):
    return {
        "attention": lambda: CausalSelfAttention(d_model, n_heads=d_model // 64, max_len=1024),
        "ssm": lambda: SelectiveSSM(d_model, d_state=64, expand=2, head_dim=64),
        "deltanet": lambda: GatedDeltaNet(d_model, n_heads=d_model // 64),
    }


def bench_layer(args, device, dtype):
    rows = []
    for name, make in mixers(args.d_model).items():
        layer = make().to(device, dtype)
        for T in args.lengths:
            x = torch.randn(args.batch, T, args.d_model, device=device, dtype=dtype, requires_grad=True)

            def fwd():
                with torch.no_grad():
                    layer(x)

            def fwd_bwd():
                layer(x).sum().backward()

            f_ms, f_mem = measure(fwd, device)
            b_ms, b_mem = measure(fwd_bwd, device)
            row = {"mixer": name, "seq_len": T, "fwd_ms": f_ms, "fwd_mem_mb": f_mem, "fwd_bwd_ms": b_ms, "fwd_bwd_mem_mb": b_mem}
            print(row, flush=True)
            rows.append(row)
    return rows


def bench_scan(args, device, dtype):
    rows = []
    H, P, N = args.d_model * 2 // 64, 64, 64
    backends = {"chunked": lambda *a: ssd_scan_chunked(*a, chunk_size=64)}
    if device.type == "cuda":
        from fadingmem.layers.triton_scan import ssd_scan_triton

        backends["triton"] = ssd_scan_triton
    for T in args.lengths:
        x = torch.randn(args.batch, T, H, P, device=device)
        a = torch.rand(args.batch, T, H, device=device)
        B, C = torch.randn(args.batch, T, N, device=device), torch.randn(args.batch, T, N, device=device)
        current = dict(backends)
        if T <= args.max_recurrent_len:
            current["recurrent"] = ssd_scan_recurrent
        for name, fn in current.items():
            with torch.no_grad():
                ms, mem = measure(lambda: fn(x, a, B, C), device, iters=5)
            row = {"backend": name, "seq_len": T, "ms": ms, "peak_mem_mb": mem}
            if name == "triton" and ms:
                bytes_moved = (x.numel() * 2 + a.numel() + B.numel() + C.numel()) * 4  # read x,a,B,C + write y
                row["achieved_GBps"] = bytes_moved / (ms * 1e-3) / 1e9
            print(row, flush=True)
            rows.append(row)
    return rows


@torch.no_grad()
def bench_decode(args, device, dtype):
    rows = []
    for name, make in mixers(args.d_model).items():
        layer = make().to(device, dtype).eval()
        max_ctx = max(args.contexts)
        cache = (layer.init_cache(args.batch, device, dtype, max_len=max_ctx + 1) if name == "attention"
                 else layer.init_cache(args.batch, device, dtype))
        x_t = torch.randn(args.batch, args.d_model, device=device, dtype=dtype)
        ctx = 0
        for target in sorted(args.contexts):
            while ctx < target:  # fill the context token by token (cheap, not timed)
                layer.step(x_t, cache)
                ctx += 1
            if name == "attention":
                def one_step():
                    cache["pos"] = target  # overwrite the same slot so the context length stays fixed
                    layer.step(x_t, cache)
                state = layer.state_bytes(args.batch, target, torch.finfo(dtype).bits // 8)
            else:
                snapshot = {k: v.clone() for k, v in cache.items()}

                def one_step():
                    layer.step(x_t, dict(snapshot))
                state = layer.state_bytes(args.batch)
            ms, _ = measure(one_step, device, warmup=5, iters=50)
            row = {"mixer": name, "context": target, "ms_per_token": ms, "state_kb": state / 1024}
            print(row, flush=True)
            rows.append(row)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="+", default=["layer", "scan", "decode"])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 2048, 8192])
    ap.add_argument("--max-recurrent-len", type=int, default=2048, help="skip the slow Python loop beyond this")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", type=Path, default=Path("results/benchmark"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    tag = torch.cuda.get_device_name(device).replace(" ", "_") if device.type == "cuda" else "cpu"
    print(f"device: {tag}, dtype: {dtype}")
    for which in args.which:
        rows = {"layer": bench_layer, "scan": bench_scan, "decode": bench_decode}[which](args, device, dtype)
        write_csv(rows, args.out / f"{which}_{tag}.csv")


if __name__ == "__main__":
    main()
