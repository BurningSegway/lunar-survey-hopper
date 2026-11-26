"""The transitions of a hop recorded by record_hop.py, and what attitude
control does at each logged step.

usage (from Sim_MuJoCo): python report_figures/simulation/scripts/hop_events.py HOP.pkl
"""
import pickle
import sys

import numpy as np


def mode(e):
    """What attitude control does: off / hold / target / level / plane."""
    if not e["airborne"]:
        return "off"
    if e["phase"] == "TAKEOFF":
        return "hold"
    if e["tracking"]:
        return "target"
    return "plane" if e["accepted"] > 0 else "level"


def events(log):
    """(index, key, text) of every transition, in order."""
    out = [(0, "jump", "Jump command")]
    prev = log[0]
    impacts = 0
    for i, e in enumerate(log[1:], 1):
        p = prev
        if e["cut_off"] and not p["cut_off"] and impacts == 0:
            out.append((i, "cutoff", "Cutoff: control on"))
        if e["ballistic"] and not p["ballistic"]:
            out.append((i, "latch", "Arc latched" if impacts == 0 else "Bounce arc latched"))
        if e["phase"] == "ASCENT" and p["phase"] != "ASCENT":
            out.append((i, "ascent", "0.5 m up: track target"))
        if e["phase"] == "DESCENT" and p["phase"] != "DESCENT":
            out.append((i, "apex", "Apex: landing fits start"))
        if e["accepted"] > 0 and p["accepted"] == 0:
            out.append((i, "fit", "First landing plane"))
        if p["tracking"] and not e["tracking"] and e["airborne"]:
            out.append((i, "swap", "Turn to the landing plane due"))
        if e["phase"] == "LANDING" and p["phase"] != "LANDING":
            out.append((i, "near", "Touchdown near"))
        if p["airborne"] and not e["airborne"]:
            impacts += 1
            out.append((i, "impact", "Impact: control off"))
        if e["airborne"] and not p["airborne"] and impacts > 0:
            out.append((i, "bounce", "Bounce: control on"))
        if e["phase"] == "GROUND" and p["phase"] != "GROUND":
            out.append((i, "ground", "Still 0.5 s: on ground"))
        prev = e
    return out


if __name__ == "__main__":
    rec = pickle.load(open(sys.argv[1], "rb"))
    log = rec["log"]
    p0 = log[0]["pos"]
    u = rec["target"][:2] - p0[:2]
    print(f"target {np.linalg.norm(u):.2f} m away")
    u = u / np.linalg.norm(u)
    for i, k, text in events(log):
        e = log[i]
        print(f"{e['t']:8.3f} s  {k:7s} s={(e['pos'][:2] - p0[:2]) @ u:6.2f} m  z={e['pos'][2]:6.2f} m  {text}")
