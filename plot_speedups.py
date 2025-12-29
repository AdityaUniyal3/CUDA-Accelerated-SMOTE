#!/usr/bin/env python3
"""
plot_speedups.py
Create two simple plots from the 'timings_paired.csv' produced by benchmark_smote.py.

USAGE:
  python plot_speedups.py --timings ./bench_runs/timings_paired.csv --N 10000 --F 1000 --minority 20
"""
import argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--N", type=int, default=None, help="Fix N for plots; if None, auto-pick first")
    ap.add_argument("--F", type=int, default=None, help="For imbalance plot, fix F")
    ap.add_argument("--minority", type=float, default=None, help="For feature plot, fix minority%")
    args = ap.parse_args()

    df = pd.read_csv(args.timings)
    if args.N is None:
        args.N = int(df["N"].iloc[0])

    # Plot A: speedup vs F at fixed N and minority%
    minority = args.minority if args.minority is not None else float(df[df["N"]==args.N]["minority_pct"].iloc[0])
    a = df[(df["N"]==args.N) & (df["minority_pct"]==minority)]
    a = a.sort_values("F")
    plt.figure()
    plt.plot(a["F"], a["speedup"], marker="o")
    plt.xlabel("Features (F)")
    plt.ylabel("Speedup (CPU sec / GPU sec)")
    plt.title(f"Speedup vs Features (N={args.N}, minority={minority}%)")
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("speedup_vs_features.png", dpi=200)

    # Plot B: speedup vs imbalance at fixed N and F
    F = args.F if args.F is not None else int(df[df["N"]==args.N]["F"].iloc[0])
    b = df[(df["N"]==args.N) & (df["F"]==F)]
    b = b.sort_values("minority_pct")
    plt.figure()
    plt.plot(b["minority_pct"], b["speedup"], marker="o")
    plt.xlabel("Minority share (%)")
    plt.ylabel("Speedup (CPU sec / GPU sec)")
    plt.title(f"Speedup vs Imbalance (N={args.N}, F={F})")
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("speedup_vs_imbalance.png", dpi=200)

    print("Saved: speedup_vs_features.png, speedup_vs_imbalance.png")

if __name__ == "__main__":
    main()
