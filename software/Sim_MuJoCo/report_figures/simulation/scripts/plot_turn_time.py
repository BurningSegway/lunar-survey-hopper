"""Turn time and seconds per degree against the step angle (flight gain, z),
with the switch's formula and single s/deg values for comparison.
usage: plot_turn_time.py SWEEP.json OUT.png   (run from Sim_MuJoCo)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, ".")
from jumper_sim.flight_reference import FlightReferenceConfig  # noqa: E402

rows = [r for r in json.load(open(sys.argv[1])) if r["gain"] == "design"]
out = Path(sys.argv[2])
KEY = "time_within_2deg_s"
LO, HI, XMAX = 10.0, 80.0, 120.0  # fitted range; plot range
config = FlightReferenceConfig()
a, k = config.turn_time_base, config.turn_time_per_degree  # the switch's formula

theta = np.array([abs(r["step_deg"]) for r in rows])
t = np.array([r[KEY] for r in rows])
keep = theta <= XMAX
theta, t = theta[keep], t[keep]
# single s/deg values: the slower direction's time over the angle
singles = {th: t[theta == th].max() / th for th in (10.0, 20.0, 60.0)}

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLES = {10.0: (AQUA, (0, (5, 3))), 20.0: (YELLOW, (0, (1.5, 2.5))), 60.0: (MAGENTA, (0, (6, 2, 1.5, 2)))}
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.0), facecolor=SURFACE)
xs = np.linspace(1.5, XMAX, 400)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.axvspan(LO, HI, color="#f0efec", lw=0, zorder=0)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("attitude error to turn away [deg]")
    ax.set_xlim(0, XMAX)
    ax.set_xticks(np.arange(0, XMAX + 1, 20))
ax1.text(0.5 * (LO + HI), 1.57, "fitted range", ha="center", va="top", color=INK2, fontsize=11)

# Left: time to within 2 deg
ax1.plot(theta, t, "o", color=BLUE, ms=7, mec=SURFACE, mew=1.3, zorder=4,
         label="measured (both directions)")
ax1.plot(xs, a + k * xs, color=ORANGE, lw=2, zorder=3,
         label=f"switch formula: t = {a:.2f} s + {k * 1000:.1f} ms/deg × angle "
               f"(s/deg = {k * 1000:.1f} ms + {a:.2f} s / angle)")
for th, s in singles.items():
    colour, style = STYLES[th]
    ax1.plot(xs, s * xs, color=colour, lw=2, ls=style, zorder=3,
             label=f"single value from {th:.0f} deg: {s:.4f} s/deg")
ax1.set_ylim(0, 1.6)
ax1.set_ylabel("time to within 2 deg [s]")
ax1.set_title("Turn time (flight LQR, z axis)", loc="left", color=INK, fontsize=13)
# where the 60 deg value promises too much: below 60 deg it is faster than the robot
th10 = 10.0
ax1.annotate(f"60 deg value: {th10:.0f} deg in {singles[60.0] * th10:.2f} s\n"
             f"(measured {t[theta == th10].max():.2f} s)",
             xy=(th10, singles[60.0] * th10), xytext=(26, 0.12), fontsize=10, color=INK2,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

# Right: seconds per degree
ax2.plot(theta, t / theta, "o", color=BLUE, ms=7, mec=SURFACE, mew=1.3, zorder=4)
ax2.plot(xs, k + a / xs, color=ORANGE, lw=2, zorder=3)
for th, s in singles.items():
    colour, style = STYLES[th]
    ax2.axhline(s, color=colour, lw=2, ls=style, zorder=3)
ax2.set_ylim(0, 0.12)
ax2.set_ylabel("seconds per degree")
ax2.set_title("Seconds per degree", loc="left", color=INK, fontsize=13)
handles, labels = ax1.get_legend_handles_labels()
order = [0, 1, 2, 3, 4]  # measured, formula, 10, 20, 60 deg
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
           ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
print("saved", out, {th: round(s, 4) for th, s in singles.items()})
