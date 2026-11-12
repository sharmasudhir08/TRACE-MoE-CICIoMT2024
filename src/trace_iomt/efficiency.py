"""Model size and inference latency (hypothesis H10)."""
import time

import numpy as np
import torch

from .models.factory import build_model, count_parameters


@torch.no_grad()
def measure_latency(model, n_features, T, device, batch_size, repeats=50, warmup=10):
    model.eval().to(device)
    x = torch.randn(batch_size, T, n_features, device=device)
    p = torch.randint(0, 4, (batch_size, T), device=device)
    for _ in range(warmup):
        model(x, p)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x, p)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = np.median(times)
    return {"ms_per_batch": 1000 * t, "ms_per_window": 1000 * t / batch_size,
            "windows_per_second": batch_size / t}


def profile_model(name, n_features, T, hierarchy, cfg_model, devices, batch_sizes=(1, 1024)):
    model = build_model(name, n_features, hierarchy, cfg_model)
    n = count_parameters(model)
    row = {"model": name, "T": T, "params": n, "size_mb_fp32": round(n * 4 / 2 ** 20, 3)}
    for dev in devices:
        for bs in batch_sizes:
            r = measure_latency(model, n_features, T, torch.device(dev), bs)
            row[f"{dev}_bs{bs}_ms_per_window"] = round(r["ms_per_window"], 4)
            row[f"{dev}_bs{bs}_windows_per_s"] = round(r["windows_per_second"], 1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row
