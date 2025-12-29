#!/usr/bin/env python3
import argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)           # timings_paired.csv
    ap.add_argument("--N", type=int, default=None)        # filter by N (recommended)
    args = ap.parse_args()

    df = pd.read_csv(args.timings)
    if args.N is not None:
        df = df[df["N"] == args.N]
    # keep only needed cols
    cols = {"N","F","minority_pct","seconds_CPU","seconds_GPU","speedup"}
    df = df[[c for c in df.columns if c in cols]].copy()

    # ----- Plot A: Speedup vs Features (one line per minority %) -----
    plt.figure(figsize=(8,6))
    for m, sub in sorted(df.groupby("minority_pct"), key=lambda x: x[0]):
        s = sub.groupby("F", as_index=False)["speedup"].median().sort_values("F")
        if len(s) >= 2:
            plt.plot(s["F"], s["speedup"], marker="o", label=f"{int(m)}%")
    plt.axhline(1.0, ls="--", alpha=0.5)
    plt.xlabel("Features (F)")
    plt.ylabel("Speedup = CPU/GPU")
    plt.title(f"Speedup vs Features" + (f" (N={args.N})" if args.N else ""))
    plt.legend(title="Minority %", ncol=3)
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("speedup_vs_features_combined.png", dpi=150)

    # ----- Plot B: Speedup vs Minority (one line per F) -----
    plt.figure(figsize=(8,6))
    for F, sub in sorted(df.groupby("F"), key=lambda x: x[0]):
        s = sub.groupby("minority_pct", as_index=False)["speedup"].median().sort_values("minority_pct")
        if len(s) >= 2:
            plt.plot(s["minority_pct"], s["speedup"], marker="o", label=f"F={F}")
    plt.axhline(1.0, ls="--", alpha=0.5)
    plt.xlabel("Minority share (%)")
    plt.ylabel("Speedup = CPU/GPU")
    plt.title(f"Speedup vs Minority%" + (f" (N={args.N})" if args.N else ""))
    plt.legend(title="Features", ncol=3)
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("speedup_vs_minority_combined.png", dpi=150)

    # ----- Plot C: Heatmap (optional but nice) -----
    try:
        # pivot to F × minority grid
        grid = (df.groupby(["F","minority_pct"])["speedup"]
                  .median().reset_index()
                  .pivot(index="F", columns="minority_pct", values="speedup")
                  .sort_index().sort_index(axis=1))
        plt.figure(figsize=(8,6))
        im = plt.imshow(grid.values, aspect="auto", origin="lower")
        plt.colorbar(im, label="Speedup (CPU/GPU)")
        plt.xticks(range(len(grid.columns)), [int(c) for c in grid.columns])
        plt.yticks(range(len(grid.index)),   [int(r) for r in grid.index])
        plt.xlabel("Minority share (%)")
        plt.ylabel("Features (F)")
        plt.title(f"Speedup Heatmap" + (f" (N={args.N})" if args.N else ""))
        plt.tight_layout()
        plt.savefig("speedup_heatmap.png", dpi=150)
    except Exception:
        pass

if __name__ == "__main__":
    main()
