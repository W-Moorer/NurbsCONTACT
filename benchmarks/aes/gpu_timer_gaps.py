"""Measure GPU timer gaps by itself, or summarize timings saved by the hinge solver."""
import argparse
import json

import numpy as np


def probe():
    import cupy as cp

    kernel = cp.RawKernel(r"""
    extern "C" __global__ void timer_gaps(unsigned long long *gaps, int *count) {
        unsigned long long start, previous, now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
        previous = start;
        int n = 0;
        do {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            if (now - previous > 100000ULL && n < 4096)
                gaps[n++] = now - previous;
            previous = now;
        } while (now - start < 400000000ULL);
        *count = n;
    }""", "timer_gaps")
    gaps = cp.empty(4096, cp.uint64)
    count = cp.empty(1, cp.int32)
    kernel((1,), (1,), (gaps, count))
    result = gaps.get()[:int(count.get()[0])] * 1e-3
    return {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "duration_ms": 400,
        "threshold_us": 100,
        "count": len(result),
        "sum_ms": float(result.sum() / 1000),
        "gap_us_p50_p95_max": np.percentile(result, [50, 95, 100]).tolist() if len(result) else [],
    }


def trace_statistics(path):
    with np.load(path) as trace:
        device = trace["step_ns"][1:]
        host = trace["host_io_step_ns"][1:]
        scan = trace["full_scan_warps"][1:]
        long_device = device > 1_000_000
        long_host = host > 1_000_000
        return {
            "steps": len(device),
            "device_steps_over_1ms": int(long_device.sum()),
            "device_steps_over_1ms_without_fallback": int((long_device & (scan == 0)).sum()),
            "host_steps_over_1ms": int(long_host.sum()),
            "host_seconds_in_steps_over_1ms": float(host[long_host].sum() * 1e-9),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", help="Read step timings from a saved response or latency NPZ")
    args = parser.parse_args()
    print(json.dumps(trace_statistics(args.response) if args.response else probe()))
