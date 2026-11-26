# Simulation figures for the report

Each figure comes as PNG (300 or 200 dpi) and PDF. The scripts that make
them are in `scripts/`; run them from `Sim_MuJoCo`.

| Figure | Shows |
|---|---|
| `hop_transitions` | One Moon hop towards a target 15.5 m away (spawn + (15, 4)) on a landscape page: side view along start -> target, cube colour = what attitude control does, arrows = body -z, the latched arcs and predicted landing spots and planes, the numbered transitions (jump command, cutoff, 0.5 m up, arc latched, apex, turn to the landing plane due, touchdown near, impacts, bounce, settling, still), and a broken-axis timeline of the hop phase and the control mode. After the bounce lands, the IMU estimate switches control on briefly while the robot rocks; those are merged into one "settling" entry. |
| `turn_time_per_degree` | The flight LQR's turn time (z axis, reference steps of 3-120 deg in both directions, time until within 2 deg) and the same as seconds per degree, with the switch's formula t = 0.59 s + 0.0039 s/deg x angle (fitted range 10-80 deg) and single s/deg values taken from 10, 20 and 60 deg. |

Regenerate:

```sh
# hop_transitions (~20 s to record, ~1 min to render)
.venv/bin/python report_figures/simulation/scripts/record_hop.py /tmp/hop_15_4.pkl 15 4
MUJOCO_GL=egl .venv/bin/python report_figures/simulation/scripts/render_hop_figure.py /tmp/hop_15_4.pkl report_figures/simulation/hop_transitions.png
.venv/bin/python report_figures/simulation/scripts/hop_events.py /tmp/hop_15_4.pkl   # the transitions as text

# turn_time_per_degree
.venv/bin/python -m benchmarks.turn_rate_benchmark --label turn_z_sweep_120 --gain design --angles 120 115 110 105 100 95 90 85 80 70 60 50 40 30 25 20 15 10 7 5 3
.venv/bin/python report_figures/simulation/scripts/plot_turn_time.py benchmarks/results/turn_z_sweep_120.json report_figures/simulation/turn_time_per_degree.png
```

Both depend on the simulation's current behaviour: rerun them after changes
to the hop process (the transitions, the switch to the landing plane) or to
the controller.
