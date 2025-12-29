#!/usr/bin/env python3
import argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)           # timings.csv (not paired)
    ap.add_argument("--N", type=int, required=True)       # fix N for both plots
    ap.add_argument("--F", type=int, default=None)        # fix F for imbalance plot
    ap.add_argument("--minority", type=float, default=None)  # fix minority% for features plot
    args = ap.parse_args()

    df = pd.read_csv(args.timings)

    # --- Plot A: runtime vs Features (at fixed N & minority %) ---
    a = df[(df["N"] == args.N)]
    if args.minority is not None:
        a = a[a["minority_pct"] == float(args.minority)]
    if not a.empty:
        cpu = a[a["impl"]=="CPU"].groupby("F")["seconds"].median().reset_index()
        gpu = a[a["impl"]=="GPU"].groupby("F")["seconds"].median().reset_index()

        plt.figure(figsize=(8,6))
        if not cpu.empty: plt.plot(cpu["F"], cpu["seconds"], marker="o", label="CPU (median sec)")
        if not gpu.empty: plt.plot(gpu["F"], gpu["seconds"], marker="o", label="GPU (median sec)")
        plt.title(f"Absolute Runtime vs Features (N={args.N}, minority={args.minority}%)")
        plt.xlabel("Features (F)"); plt.ylabel("Seconds (median)"); plt.legend(); plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig("runtime_vs_features.png", dpi=150)

    # --- Plot B: runtime vs Imbalance (at fixed N & F) ---
    if args.F is not None:
        b = df[(df["N"] == args.N) & (df["F"] == args.F)]
        if not b.empty:
            cpu = b[b["impl"]=="CPU"].groupby("minority_pct")["seconds"].median().reset_index()
            gpu = b[b["impl"]=="GPU"].groupby("minority_pct")["seconds"].median().reset_index()

            plt.figure(figsize=(8,6))
            if not cpu.empty: plt.plot(cpu["minority_pct"], cpu["seconds"], marker="o", label="CPU (median sec)")
            if not gpu.empty: plt.plot(gpu["minority_pct"], gpu["seconds"], marker="o", label="GPU (median sec)")
            plt.title(f"Absolute Runtime vs Imbalance (N={args.N}, F={args.F})")
            plt.xlabel("Minority share (%)"); plt.ylabel("Seconds (median)"); plt.legend(); plt.grid(True, ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig("runtime_vs_imbalance.png", dpi=150)

if __name__ == "__main__":
    main()
