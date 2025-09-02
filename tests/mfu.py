import time, re, torch

def _pick_dim(total_mem_bytes, dtype=torch.float16):
    """Choose a square size (M=N=K) that fits comfortably in memory."""
    if total_mem_bytes is None:  # CPU heuristic
        return 2048
    gb = total_mem_bytes / (1024**3)
    if gb < 6:    return 2048
    if gb < 10:   return 4096
    if gb < 16:   return 6144
    if gb < 24:   return 8192
    return 12288

def measure_gemm_tflops(M=None, N=None, K=None, *, dtype=torch.float32,
                        device=None, iters=50, warmup=5, allow_tf32=False):
    """
    Measure effective TFLOPS via GEMM: C = A @ B
    FLOPs = 2 * M * N * K per matmul.
    By default we DISABLE TF32 so fp32 is 'strict'.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory
        dev_name = torch.cuda.get_device_name(0)
    else:
        total_mem = None
        dev_name = "CPU"

    if M is None or N is None or K is None:
        dim = _pick_dim(total_mem, dtype)
        M = N = K = dim

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    if device == "cuda":
        for _ in range(warmup):
            _ = a @ b
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            _ = a @ b

    # Timed runs
    t0 = time.perf_counter()
    if device == "cuda":
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
    else:
        for _ in range(iters):
            _ = a @ b
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / iters
    flops = 2 * M * N * K
    tflops = flops / avg_s / 1e12

    return {
        "device": device,
        "device_name": dev_name,
        "dtype": str(dtype).split(".")[-1],
        "dims": (M, N, K),
        "avg_time_s": avg_s,
        "tflops": tflops,
        "tf32": bool(allow_tf32),
    }

def _guess_promised_tflops(device_name: str):
    """
    Approx theoretical TFLOPS for common Colab GPUs.
    Returns dict with keys 'fp32' and 'bf16' (None if unknown/unsupported).
    """
    dn = device_name.lower()
    table = [
        (r"\bt4\b",     8.1,   None),   # Tesla T4
        (r"\bp100\b",   9.3,   None),   # Tesla P100
        (r"\bv100\b",  14.0,   None),   # V100 (no native bf16)
        (r"\ba100\b",  19.5,  312.0),   # A100 (bf16 tensor core)
        (r"\ba10g?\b", 31.2,  124.0),   # A10/A10G
        (r"\bl4\b",    30.3,  242.0),   # L4 approx
    ]
    for pattern, fp32, bf16 in table:
        if re.search(pattern, dn):
            return {"fp32": fp32, "bf16": bf16}
    return {"fp32": None, "bf16": None}

def run_perf_and_mfu(*, promised_fp32_tflops=None, promised_bf16_tflops=None,
                     iters=40, warmup=5):
    """
    Measure FP32 (strict) and BF16 GEMM TFLOPS and compute MFU = measured / promised.
    If promised numbers aren't provided, tries to infer them from GPU name.
    Returns a dict with results for 'fp32' and 'bf16'.
    """
    dev =  "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"Device: {dev_name} ({dev})")

    auto = _guess_promised_tflops(dev_name) if dev == "cuda" else {"fp32": None, "bf16": None}
    promised_fp32 = promised_fp32_tflops if promised_fp32_tflops is not None else auto["fp32"]
    promised_bf16 = promised_bf16_tflops if promised_bf16_tflops is not None else auto["bf16"]

    results = {}

    # FP32 (strict — TF32 disabled)
    r32 = measure_gemm_tflops(dtype=torch.float32, device=dev, iters=iters,
                              warmup=warmup, allow_tf32=False)
    mfu32 = (r32["tflops"] / promised_fp32) if (promised_fp32 and promised_fp32 > 0) else None
    results["fp32"] = {
        "measured_tflops": r32["tflops"],
        "promised_tflops": promised_fp32,
        "mfu": mfu32,
        "dims": r32["dims"],
        "avg_ms": r32["avg_time_s"] * 1e3,
    }

    # BF16 (if supported)
    if dev == "cuda" and torch.cuda.is_bf16_supported():
        rb = measure_gemm_tflops(dtype=torch.bfloat16, device=dev, iters=iters,
                                 warmup=warmup, allow_tf32=False)
        mfub = (rb["tflops"] / promised_bf16) if (promised_bf16 and promised_bf16 > 0) else None
        results["bf16"] = {
            "measured_tflops": rb["tflops"],
            "promised_tflops": promised_bf16,
            "mfu": mfub,
            "dims": rb["dims"],
            "avg_ms": rb["avg_time_s"] * 1e3,
        }
    else:
        results["bf16"] = {"note": "BF16 not supported on this device or CUDA not available."}

    # Pretty print
    def _fmt(label, res):
        if "note" in res:
            print(f"{label}: {res['note']}")
            return
        t = res["measured_tflops"]
        prom = res["promised_tflops"]
        mfu = res["mfu"]
        M, N, K = res["dims"]
        msg = f"{label}: {t:.2f} TFLOPS (dims={M}x{K}x{N}, avg={res['avg_ms']:.2f} ms)"
        if prom:
            msg += f" | promised {prom:.1f} TFLOPS | MFU={mfu*100:.1f}%"
        else:
            msg += " | promised TFLOPS: unknown"
        print(msg)

    _fmt("FP32", results["fp32"])
    _fmt("BF16", results["bf16"])

    if dev == "cuda":
        missing = []
        if results["fp32"]["promised_tflops"] is None:
            missing.append("fp32")
        if isinstance(results["bf16"], dict) and results["bf16"].get("promised_tflops") is None and "note" not in results["bf16"]:
            missing.append("bf16")
        if missing:
            print("\nNote: Promised TFLOPS not recognized for this GPU. "
                  "Pass them explicitly, e.g.:")
            print("run_perf_and_mfu(promised_fp32_tflops=19.5, promised_bf16_tflops=312.0)  # A100 example")
    return results

# Example:
results = run_perf_and_mfu()
# results = run_perf_and_mfu(promised_fp32_tflops=19.5, promised_bf16_tflops=312.0)  # override
