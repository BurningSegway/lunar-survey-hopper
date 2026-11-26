"""Hop figure for a landscape page: side view along start -> target with the
cubes coloured by what attitude control does, the numbered transitions, and
a broken-axis timeline.

usage (from Sim_MuJoCo):
    MUJOCO_GL=egl python report_figures/simulation/scripts/render_hop_figure.py HOP.pkl OUT.png
HOP.pkl from record_hop.py; also writes OUT.pdf.
"""
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hop_events import events, mode  # noqa: E402

rec = pickle.load(open(sys.argv[1], "rb"))
OUT = Path(sys.argv[2])
log = rec["log"]
G = rec["gravity"] if rec["gravity"] > 0 else 1.62  # (older recordings stored 0)
FS = 13.5  # base font size [pt]

MODES = {
    "off": ("Control off (on the ground)", (0.30, 0.30, 0.30)),
    "hold": ("Holds the pre-jump attitude", (0.78, 0.78, 0.78)),
    "target": ("Body $-z$ tracks the target", (1.00, 0.45, 0.00)),
    "level": ("Level (no plane yet)", (0.00, 0.62, 0.45)),
    "plane": ("Aligns with the landing plane", (0.00, 0.45, 0.70)),
}
TEXT = {
    "jump": "Jump command",
    "cutoff": "Cutoff: control on",
    "ascent": "0.5 m up: track the target",
    "latch": "Cutoff + 60 ms: arc latched",
    "apex": "Apex: spot and plane fits start",
    "swap": "Turn to the landing plane due",
    "near": "Touchdown near: LANDING",
    "impact": "Impact: control off",
    "bounce": "Bounce: control on, new arc",
    "settle": "Settling: control briefly on (rocking)",
    "ground": "Still for 0.5 s: GROUND",
}
raw = events(log)
cutoff_i = next(i for i, k, _ in raw if k == "cutoff")
raw_impacts = [i for i, k, _ in raw if k == "impact"]
raw_bounces = [i for i, k, _ in raw if k == "bounce"]
EV = []
for i, k, _ in raw:
    if k == "fit":
        continue  # the first fit is on the apex sample
    if k == "latch" and i > cutoff_i + 200:
        continue  # a bounce's arc latches on its control-on sample
    if len(raw_impacts) > 1 and i > raw_impacts[1] and k in ("bounce", "impact"):
        # after the bounce has landed: the robot settles; one entry for it
        if not EV or EV[-1][1] != "settle":
            EV.append((i, "settle"))
        continue
    EV.append((i, k))
numbered = list(enumerate(EV, 1))
impacts = [i for i, k in EV if k == "impact"]
bounces = [i for i, k in EV if k == "bounce"]
first_impact = impacts[0]
last_impact = raw_impacts[-1]

# --------------------------------------------------- view: along start -> target
pos = np.array([e["pos"] for e in log])
target = rec["target"]
start_xy = pos[0][:2]
u = target[:2] - start_xy
target_s = float(np.linalg.norm(u))
u = u / target_s
S = (pos[:, :2] - start_xy) @ u  # along-track coordinate
Zp = pos[:, 2]
X0, X1, Z0, Z1 = -1.6, target_s + 1.2, -0.8, 17.2

H = 4000
W = int(round(H * (X1 - X0) / (Z1 - Z0)))
model = mujoco.MjModel.from_xml_path("build/Jumping_Robot.xml")
model.vis.global_.offwidth, model.vis.global_.offheight = W, H
model.vis.global_.orthographic = 1  # (MuJoCo 3.2.3 ignores MjvCamera.orthographic)
model.vis.global_.fovy = Z1 - Z0  # orthographic: the view height in m
model.vis.quality.shadowsize = 0
data = mujoco.MjData(model)
box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_box")
target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object")
model.geom_pos[target_geom] = (target[0], target[1], target[2] - 0.15)

# Copies: every ~0.75 m along the path in flight (0.45 m in the bounce),
# plus one per transition unless one is already close by.
keep, travelled = [], 0.0
event_idx = {i for i, _ in EV}
for i in range(1, len(log)):
    travelled += float(np.linalg.norm(pos[i] - pos[i - 1]))
    spacing = 0.75 if i < first_impact else 0.45
    if i in event_idx or not log[i]["airborne"] or i >= last_impact:
        continue
    if travelled >= spacing:
        keep.append(i)
        travelled = 0.0
for i, k in EV:
    if not any(np.linalg.norm(pos[i] - pos[j]) < 0.4 for j, _ in EV if j in keep):
        keep = [j for j in keep if np.linalg.norm(pos[i] - pos[j]) > 0.35]
        keep.append(i)
keep.sort()

heading = math.degrees(math.atan2(u[1], u[0]))
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
mid_s = 0.5 * (X0 + X1)
cam.lookat[:] = (*(start_xy + mid_s * u), 0.5 * (Z0 + Z1))
cam.distance, cam.azimuth, cam.elevation = 40.0, heading + 90.0, 0.0
opt = mujoco.MjvOption()
opt.geomgroup[2] = 0  # no terrain: the ground is drawn as a profile
renderer = mujoco.Renderer(model, H, W)


def load(i, d):
    d.qpos[:] = log[i]["qpos"]
    mujoco.mj_forward(model, d)


load(keep[0], data)
renderer.update_scene(data, cam, opt)
scene = renderer.scene
for i in keep[1:]:
    d = mujoco.MjData(model)
    load(i, d)
    mujoco.mjv_addGeoms(model, d, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
n = 0
for g in scene.geoms[: scene.ngeom]:
    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == box:
        g.rgba[:] = (*MODES[mode(log[keep[n]])][1], 1.0)
        n += 1
assert n == len(keep)
rgb = renderer.render()
renderer.enable_depth_rendering()  # background = the far plane (transparent)
depth = renderer.render()
renderer.disable_depth_rendering()
rgba = np.dstack([rgb, np.where(depth < 0.999 * depth.max(), 255, 0).astype(np.uint8)])

# Ground profile along the view line (start -> target).
d = mujoco.MjData(model)
mujoco.mj_forward(model, d)
groups = np.zeros(6, np.uint8)
groups[2] = 1
gid = np.zeros(1, np.int32)
ss = np.linspace(X0, X1, 800)
ground = []
for s in ss:
    xy = start_xy + s * u
    dist = mujoco.mj_ray(model, d, np.array([xy[0], xy[1], 50.0]), np.array([0.0, 0.0, -1.0]),
                         groups, 1, -1, gid)
    ground.append(50.0 - dist if dist >= 0 else np.nan)
ground = np.array(ground)


def arc_points(latch_i, end_i):
    t_arc, pa, va = log[latch_i]["arc"]
    ts = np.linspace(t_arc, log[end_i]["t"], 300) - t_arc
    p = pa[None, :] + va[None, :] * ts[:, None] - 0.5 * G * ts[:, None] ** 2 * np.array([0, 0, 1.0])
    return (p[:, :2] - start_xy) @ u, p[:, 2]


# arcs: the jump's (latched at 4) and each long bounce's, to their impacts;
# the predicted spot and plane: the last fit before those impacts
arcs = [arc_points(next(i for i, k in EV if k == "latch"), first_impact)]
flights_ends = [first_impact]
for b in raw_bounces:
    end = next(i for i in raw_impacts if i > b)
    if log[end]["t"] - log[b]["t"] > 1.0:
        arcs.append(arc_points(next(j for j in range(b, end) if log[j]["arc"] is not None), end))
        flights_ends.append(end)
spots = []
for end in flights_ends:
    anchor, normal, _ = log[end - 1]["spot"]
    spots.append((float((anchor[:2] - start_xy) @ u), float(anchor[2]),
                  float(normal[:2] @ u), float(normal[2])))


# ---------------------------------------------------------------- drawing
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "axes.linewidth": 0.9})


def badge(axis, x, y, number, size=FS - 1.5, **kw):
    axis.text(x, y, str(number), ha="center", va="center", fontsize=size, fontweight="bold",
              bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="k", lw=1.0), zorder=8, **kw)


def scene_axis(axis, arrow, arrow_lw):
    axis.fill_between(ss, ground, Z0 - 5, color=(0.87, 0.85, 0.80), lw=0, zorder=0)
    axis.plot(ss, ground, color=(0.55, 0.52, 0.45), lw=1.0, zorder=0)
    axis.plot(S[: last_impact + 1], Zp[: last_impact + 1], color="0.45", lw=0.9, zorder=1)
    for a_s, a_z in arcs:
        axis.plot(a_s, a_z, color="k", lw=1.1, ls=(0, (3, 3)), zorder=3)
    axis.imshow(rgba, extent=(X0, X1, Z0, Z1), interpolation="lanczos", zorder=2)
    for i in keep:
        if not log[i]["airborne"]:
            continue
        w, x, y, z = log[i]["qpos"][3:7]
        down = -np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        ds, dz = float(down[:2] @ u), float(down[2])
        axis.annotate("", xy=(S[i] + arrow * ds, Zp[i] + arrow * dz), xytext=(S[i], Zp[i]),
                      arrowprops=dict(arrowstyle="-|>", lw=arrow_lw, color="k", shrinkA=0, shrinkB=0,
                                      mutation_scale=8), zorder=4)
    for s, z, ns, nz in spots:
        t = np.array([nz, -ns]) / math.hypot(ns, nz)
        pts = np.array([(s, z) - 0.9 * t, (s, z) + 0.9 * t])
        axis.plot(pts[:, 0], pts[:, 1], color=MODES["plane"][1], lw=3.0, zorder=3, solid_capstyle="butt")
        axis.plot(s, z, "x", color=MODES["plane"][1], ms=8, mew=2.2, zorder=3)
    axis.set_aspect("equal")


def leader(axis, i, xl, zl, n):
    axis.plot([S[i], xl], [Zp[i], zl], color="k", lw=0.7, zorder=6)
    axis.plot(S[i], Zp[i], "o", color="k", ms=3, zorder=6)
    badge(axis, xl, zl, n)


fig = plt.figure(figsize=(11.0, 7.2))
FW, FH = fig.get_size_inches()
main_b, main_h = 0.35, 0.635
main_w = main_h * FH / FW * (X1 - X0) / (Z1 - Z0)
ax = fig.add_axes([0.055, main_b, main_w, main_h])
scene_axis(ax, 0.95, 1.0)
ax.set_xlim(X0, X1)
ax.set_ylim(Z0, Z1)
ax.set_xlabel("distance towards the target [m]")
ax.set_ylabel("height [m]")
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("Target", xy=(target_s, target[2] + 0.4), xytext=(target_s, target[2] + 2.0),
            ha="center", color=MODES["target"][1], fontsize=FS - 0.5,
            arrowprops=dict(arrowstyle="-|>", color=MODES["target"][1], lw=1.0, mutation_scale=10))

# take-off cluster: a stack on the left; apex and swap: single labels;
# touchdown and the bounce's landing: stacks above them
take_off = [(n, i) for n, (i, k) in numbered if k in ("jump", "cutoff", "ascent", "latch")]
for slot, (n, i) in enumerate(sorted(take_off, key=lambda g: Zp[g[1]])):
    leader(ax, i, -0.95, 1.4 + slot * 1.25, n)
for n, (i, k) in numbered:
    if k == "apex":
        leader(ax, i, S[i], Zp[i] + 0.0001, n)  # on the apex itself
    elif k == "swap":
        leader(ax, i, S[i] - 1.7, Zp[i] - 0.6, n)
landing_ev = [(n, i) for n, (i, k) in numbered if k in ("near", "impact", "bounce", "settle", "ground")]
near_group = [(n, i) for n, i in landing_ev if i <= bounces[0]]
end_group = [(n, i) for n, i in landing_ev if i > bounces[0]]
for group, xl, z_from in ((near_group, S[first_impact] + 0.9, 3.2),
                          (end_group, S[last_impact] + 1.4, 2.4)):
    for slot, (n, i) in enumerate(sorted(group, key=lambda g: Zp[g[1]])):
        leader(ax, i, xl, z_from + slot * 1.25, n)

# Transitions list, right of the main panel.
lx = 0.055 + main_w + 0.035
lg = fig.add_axes([lx, main_b, 0.995 - lx, main_h])
lg.axis("off")
lg.set_xlim(0, 1)
lg.set_ylim(0, 1)
lg.text(0, 1.0, "Transitions", fontweight="bold", va="top", fontsize=FS + 0.5)
row = 0.93 / max(len(numbered), 11)
y = 0.93 - 0.5 * row
for n, (i, k) in numbered:
    t = log[i]["t"]
    badge(lg, 0.035, y, n)
    lg.text(0.235, y, f"{t:.3f} s" if t < 1 else f"{t:.2f} s", ha="right", va="center", color="0.35")
    lg.text(0.27, y, TEXT[k], va="center")
    y -= row

# Legend row: cube colours and line styles.
used = [m for m in MODES if m in {mode(e) for e in log}]
handles = [plt.Rectangle((0, 0), 1, 1, fc=MODES[m][1], ec="k", lw=0.6) for m in used]
labels = [MODES[m][0] for m in used]
handles += [plt.Line2D([], [], color="0.45", lw=1.2), plt.Line2D([], [], color="k", lw=1.2, ls=(0, (3, 3))),
            plt.Line2D([], [], color=MODES["plane"][1], lw=3, marker="x", ms=8, mew=2)]
labels += ["True path", "Latched arc", "Predicted spot and plane"]
handles.insert(len(used), plt.Line2D([], [], color="k", lw=0, marker=r"$\downarrow$", ms=14))
labels.insert(len(used), "Body $-z$")
fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.235), ncol=4, frameon=False,
           fontsize=FS - 1.5, handlelength=1.6, columnspacing=1.4, handletextpad=0.5)

# ---------------------------------------------------------------- timeline
times = np.array([e["t"] for e in log])
modes = [mode(e) for e in log]
phases = [e["phase"] for e in log]


def runs(values):
    out, s0 = [], 0
    for j in range(1, len(values) + 1):
        if j == len(values) or values[j] != values[s0]:
            out.append((times[s0] - 0.001, times[j - 1], values[s0]))
            s0 = j
    return out


# broken axis: take-off, flight, touchdown, the bounce, its landing and the settling, the rest
near_t = times[next(i for i, k in EV if k == "near")]
landing2 = times[impacts[1]]
settle = [i for i, k in EV if k == "settle"]
end2 = (times[raw_impacts[-1]] if settle else landing2) + 0.1
segments = [(-0.004, 0.125), (0.125, near_t - 0.1), (near_t - 0.1, times[bounces[0]] + 0.1),
            (times[bounces[0]] + 0.1, landing2 - 0.1), (landing2 - 0.1, end2), (end2, times[-1] + 0.1)]
widths = np.array([0.19, 0.19, 0.18, 0.1, 0.2, 0.14])
widths /= widths.sum()
x_left, total, gap = 0.085, 0.905, 0.012
starts = x_left + np.concatenate([[0], np.cumsum(widths[:-1])]) * total
axes = []
for (a, b), x0, wf in zip(segments, starts, widths):
    tax = fig.add_axes([x0, 0.06, wf * total - gap, 0.105])
    axes.append(tax)
    for t_a, t_b, m in runs(modes):
        tax.barh(0.5, t_b - t_a, left=t_a, height=0.9, color=MODES[m][1], lw=0)
    for t_a, t_b, ph in runs(phases):
        tax.barh(1.55, t_b - t_a, left=t_a, height=0.9, color="0.94", ec="0.55", lw=0.6)
        lo, hi = max(t_a, a), min(t_b, b)
        if hi - lo > 0.45 * (b - a) * (0.18 / wf if wf < 0.18 else 1):
            tax.text(0.5 * (lo + hi), 1.55, ph, ha="center", va="center", fontsize=FS - 3.5, clip_on=True)
    for n, (i, k) in numbered:
        if a <= times[i] <= b:
            tax.axvline(times[i], color="k", lw=0.9, ymin=0, ymax=2.05 / 2.6)
            badge(tax, times[i], 2.45, n, size=FS - 3, clip_on=False)
    tax.set_xlim(a, b)
    tax.set_ylim(0, 2.6)
    tax.set_yticks([])
    for side in ("top", "right", "left"):
        tax.spines[side].set_visible(False)
    tax.tick_params(axis="x", labelsize=FS - 3.5)
    tax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
axes[0].set_yticks([0.5, 1.55])
axes[0].set_yticklabels(["Control", "Phase"], fontsize=FS - 2.5)
axes[0].tick_params(axis="y", length=0)
for tax in axes[1:]:
    tax.text(-0.012 / (tax.get_position().width), 0.0, "//", transform=tax.transAxes, ha="center",
             va="center", fontsize=FS - 3)
fig.text(0.53, 0.008, "time since the jump command [s] (broken axis)", ha="center", fontsize=FS - 2)

fig.savefig(OUT, dpi=300)
fig.savefig(OUT.with_suffix(".pdf"), dpi=300)
print("saved", OUT, "copies", len(keep))
for n, (i, k) in numbered:
    print(n, k, round(times[i], 3), round(S[i], 2), round(Zp[i], 2))
