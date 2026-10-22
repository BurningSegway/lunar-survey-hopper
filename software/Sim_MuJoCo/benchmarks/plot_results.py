"""Plot saved benchmark results side by side.

Usage:
    python -m benchmarks.plot_results baseline fix1_inertia [more labels...]
Writes benchmarks/results/compare_<labels>.png
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .lqr_benchmark import RESULTS_DIR, SETTLE_THRESHOLD_DEG


def main(labels: list[str]) -> None:
    runs = {label: json.loads((RESULTS_DIR / f"{label}.json").read_text())
            for label in labels}
    scenarios = []
    for results in runs.values():
        for r in results:
            if r["scenario"] not in scenarios:
                scenarios.append(r["scenario"])
    cols = 2 if len(scenarios) <= 4 else 3
    rows = -(-len(scenarios) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows),
                             sharex=True, squeeze=False)
    colors = {label: f"C{i}" for i, label in enumerate(runs)}
    for ax, name in zip(axes.flat, scenarios):
        for label, results in runs.items():
            match = [r for r in results if r["scenario"] == name]
            if match:
                tr = match[0]["trace"]
                ax.plot(tr["t"], tr["err"], label=label, color=colors[label],
                        linewidth=1.6)
        ax.axhline(SETTLE_THRESHOLD_DEG, color="0.6", linestyle=":", linewidth=1)
        ax.set_title(name)
        ax.set_ylabel("attitude error [deg]")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time in flight [s]")
    handles = [plt.Line2D([], [], color=colors[label], linewidth=1.6) for label in runs]
    fig.legend(handles, list(runs), loc="lower right", bbox_to_anchor=(0.98, 0.04))
    fig.tight_layout()
    out = RESULTS_DIR / f"compare_{'_vs_'.join(labels)}.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
