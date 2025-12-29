#!/usr/bin/env python3
"""
benchmark_smote.py  — CPU vs GPU benchmarking harness for your SMOTE/SMOGN tool.

What it does
- Generates synthetic CSV datasets (Gaussian blobs).
- Runs BOTH CPU and GPU binaries on each (N, F, minority%, k) combination.
- Records wall-time + (best-effort) peak RAM/VRAM into timings.csv (append-safe).
- Writes timings_paired.csv with seconds_CPU/seconds_GPU + speedup.

New features
- --skip_existing   → Skips combos that already have CPU+GPU rows in timings.csv
- --no_write        → Sends --out to NUL (/dev/null) so no heavy output files
- --pair_only       → Only (re)build timings_paired.csv from timings.csv and exit

Example
  python benchmark_smote.py ^
    --cpu_exe .\smote_cpu.exe ^
    --gpu_exe .\smote_gpu.exe ^
    --N 10000 ^
    --features 10,50,100,200,500,1000 ^
    --minority 1,2,5,10,20,40 ^
    --k 5 ^
    --reps 1 ^
    --no_write ^
    --skip_existing ^
    --workdir .\bench_grid
"""

import argparse, os, time, csv, random, sys
from pathlib import Path
from statistics import median

# Optional dependencies for monitoring (best-effort)
try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

# ---- Adjust if your tools use different flags ----
CPU_CMD_FMT = "{exe} --in {inp} --out {out} --k {k} --seed {seed}"
GPU_CMD_FMT = "{exe} --in {inp} --out {out} --k {k} --seed {seed}"


def gen_dataset_csv(path, N, F, minority_pct, seed=42):
    """
    Create a CSV with N rows, F numeric features, last column = label (0/1).
    minority_pct is percentage (0-100) of label=1 before augmentation.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_min = int(round(N * (minority_pct / 100.0)))
    n_maj = N - n_min

    X0 = rng.normal(loc=0.0, scale=1.0, size=(n_maj, F))
    X1 = rng.normal(loc=1.0, scale=1.0, size=(n_min, F))

    df0 = pd.DataFrame(X0); df0["label"] = 0
    df1 = pd.DataFrame(X1); df1["label"] = 1
    df = pd.concat([df0, df1], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)


def monitor_process(proc, poll_sec=0.05, track_gpu=False, gpu_index=0):
    """
    Poll a running process for wall time and (peak) RAM / VRAM usage.
    Returns (seconds, peak_rss_gb, peak_vram_gb).
    """
    start = time.perf_counter()
    peak_rss = 0
    peak_vram = 0

    p = psutil.Process(proc.pid) if psutil is not None else None

    while True:
        ret = proc.poll()
        # CPU RAM (RSS)
        if p is not None:
            try:
                rss = p.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass
        # GPU VRAM (best-effort)
        if track_gpu and _NVML_OK:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                if mem > peak_vram:
                    peak_vram = mem
            except Exception:
                pass
        if ret is not None:
            break
        time.sleep(poll_sec)

    secs = time.perf_counter() - start
    return secs, (peak_rss / (1024**3)), (peak_vram / (1024**3))


def run_once(cmd, track_gpu=False, cwd=None):
    import subprocess
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd)
    secs, gb_ram, gb_vram = monitor_process(proc, track_gpu=track_gpu)
    return secs, gb_ram, gb_vram, proc.returncode


# -------------------- Skip-existing helpers --------------------

def load_done_map(workdir):
    """Return a set of (N,F,m,k,impl) already present in timings.csv, if it exists."""
    done = set()
    try:
        import pandas as pd
        t = Path(workdir) / "timings.csv"
        if t.exists():
            df = pd.read_csv(t)
            for _, row in df.iterrows():
                done.add((int(row["N"]), int(row["F"]), float(row["minority_pct"]), int(row["k"]), str(row["impl"])))
    except Exception:
        pass
    return done


def already_paired(workdir, N, F, m, k):
    """True if timings.csv already has BOTH CPU and GPU rows for this combo."""
    try:
        import pandas as pd
        t = Path(workdir) / "timings.csv"
        if not t.exists():
            return False
        df = pd.read_csv(t)
        sub = df[(df["N"] == N) & (df["F"] == F) & (df["minority_pct"] == m) & (df["k"] == k)]
        impls = set(sub["impl"].astype(str))
        return {"CPU", "GPU"}.issubset(impls)
    except Exception:
        return False


# -------------------- Pair-file writer (fast merge) --------------------

def write_paired(workdir):
    """Build timings_paired.csv from timings.csv using a fast merge (no MultiIndex pivot)."""
    import pandas as pd

    work = Path(workdir)
    t = work / "timings.csv"
    if not t.exists():
        print(f"[pair] {t} not found")
        return
    df = pd.read_csv(t)
    if df.empty:
        print("[pair] timings.csv is empty")
        return

    cpu = df[df['impl'] == 'CPU'][['N', 'F', 'minority_pct', 'k', 'seconds',
                                   'peak_ram_gb', 'peak_vram_gb']].rename(
        columns={'seconds': 'seconds_CPU',
                 'peak_ram_gb': 'peak_ram_gb_CPU',
                 'peak_vram_gb': 'peak_vram_gb_CPU'})

    gpu = df[df['impl'] == 'GPU'][['N', 'F', 'minority_pct', 'k', 'seconds',
                                   'peak_ram_gb', 'peak_vram_gb']].rename(
        columns={'seconds': 'seconds_GPU',
                 'peak_ram_gb': 'peak_ram_gb_GPU',
                 'peak_vram_gb': 'peak_vram_gb_GPU'})

    merged = pd.merge(cpu, gpu, on=['N', 'F', 'minority_pct', 'k'], how='inner')
    if not merged.empty:
        merged['speedup'] = merged['seconds_CPU'] / merged['seconds_GPU']
    out = work / "timings_paired.csv"
    merged.to_csv(out, index=False)
    print(f"[pair] wrote {out} (rows={len(merged)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_exe", required=True, help="Path to CPU binary")
    ap.add_argument("--gpu_exe", required=True, help="Path to GPU binary")
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--features", type=str, default="10,100,1000,10000")
    ap.add_argument("--minority", type=str, default="40,20,5,1", help="CSV list of percentages")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3, help="repeat each run and take median")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workdir", type=str, default="./bench_runs")
    ap.add_argument("--gpu_index", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip (N,F,minority,k) combos that already have CPU and GPU rows in timings.csv")
    ap.add_argument("--no_write", action="store_true",
                    help="Send --out to OS null device (NUL or /dev/null) to avoid writing output files")
    ap.add_argument("--pair_only", action="store_true",
                    help="Only build timings_paired.csv from timings.csv and exit")
    args = ap.parse_args()

    if args.pair_only:
        write_paired(args.workdir)
        sys.exit(0)

    features = [int(x.strip()) for x in args.features.split(",") if x.strip()]
    minorities = [float(x.strip()) for x in args.minority.split(",") if x.strip()]

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    # Prepare timings.csv in append-safe way
    results_path = work / "timings.csv"
    write_header = not results_path.exists()
    f = open(results_path, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["N", "F", "minority_pct", "k", "impl", "seconds", "peak_ram_gb", "peak_vram_gb"])

    # Decide OS null path when --no_write
    NULL_DEV = "NUL" if os.name == "nt" else "/dev/null"

    # Optional: load existing map (not strictly required, but cheap)
    _done = load_done_map(args.workdir)

    try:
        for F in features:
            for m in minorities:
                if args.skip_existing and already_paired(args.workdir, args.N, F, float(m), args.k):
                    print(f"Skipping F={F}, minority={m}% (CPU+GPU timings already exist)")
                    continue

                data_path = work / f"data_N{args.N}_F{F}_m{int(m)}.csv"
                if not data_path.exists():
                    print(f"[gen] {data_path.name}")
                    gen_dataset_csv(data_path, args.N, F, m, seed=args.seed)
                else:
                    print(f"[gen] exists: {data_path.name}")

                # --- CPU run(s) ---
                cpu_secs, cpu_ram_gb, cpu_vram_gb = [], [], []
                for r in range(args.reps):
                    out_cpu = NULL_DEV if args.no_write else str(work / f"cpu_out_N{args.N}_F{F}_m{int(m)}_r{r}.csv")
                    cmd = CPU_CMD_FMT.format(exe=args.cpu_exe, inp=str(data_path), out=out_cpu, k=args.k, seed=args.seed)
                    secs, ram, vram, rc = run_once(cmd, track_gpu=False)
                    if rc != 0:
                        print(f"[CPU] Non-zero exit code {rc} for F={F}, m={m}", file=sys.stderr)
                    cpu_secs.append(secs); cpu_ram_gb.append(ram); cpu_vram_gb.append(vram)
                w.writerow([args.N, F, m, args.k, "CPU", median(cpu_secs), max(cpu_ram_gb), max(cpu_vram_gb)])
                f.flush()

                # --- GPU run(s) ---
                gpu_secs, gpu_ram_gb, gpu_vram_gb = [], [], []
                for r in range(args.reps):
                    out_gpu = NULL_DEV if args.no_write else str(work / f"gpu_out_N{args.N}_F{F}_m{int(m)}_r{r}.csv")
                    cmd = GPU_CMD_FMT.format(exe=args.gpu_exe, inp=str(data_path), out=out_gpu, k=args.k, seed=args.seed)
                    secs, ram, vram, rc = run_once(cmd, track_gpu=True, cwd=None)
                    if rc != 0:
                        print(f"[GPU] Non-zero exit code {rc} for F={F}, m={m}", file=sys.stderr)
                    gpu_secs.append(secs); gpu_ram_gb.append(ram); gpu_vram_gb.append(vram)
                w.writerow([args.N, F, m, args.k, "GPU", median(gpu_secs), max(gpu_ram_gb), max(gpu_vram_gb)])
                f.flush()

    finally:
        f.close()

    # Build paired file at the end (fast)
    write_paired(args.workdir)
    print(f"\nWrote results to:\n  {results_path}\n  {work/'timings_paired.csv'}")
    print("Tip: plot speedups with:\n  python plot_speedups.py --timings " +
          str(work / 'timings_paired.csv') + f" --N {args.N} --F {features[-1]} --minority {minorities[0]}")


if __name__ == "__main__":
    main()
