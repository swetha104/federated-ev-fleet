"""Build the README figures in docs/media/ from data/ and outputs/.

    python scripts/figures.py

Colours follow a validated categorical palette: three slots for scatter plots
(every pair stays distinguishable under colour-vision deficiency), up to four
for line charts. The seven EV models are grouped into three battery classes for
that reason, which is also the physically meaningful grouping.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evfl.data import EVTask, ev_windows, load_ev

OUT = Path("outputs")
MEDIA = Path("docs/media")

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order

CAPACITY_TO_MODEL = {32300: "VW e-Up", 39000: "BMW i3", 58000: "VW ID.3", 64000: "Kia Soul EV",
                     77000: "VW ID.4", 77400: "Kia EV6 GT", 80000: "SUV"}
MODEL_TO_CLASS = {"VW e-Up": "small (32–39 kWh)", "BMW i3": "small (32–39 kWh)",
                  "VW ID.3": "mid (58–64 kWh)", "Kia Soul EV": "mid (58–64 kWh)",
                  "VW ID.4": "large (77–80 kWh)", "Kia EV6 GT": "large (77–80 kWh)", "SUV": "large (77–80 kWh)"}
CLASSES = ["small (32–39 kWh)", "mid (58–64 kWh)", "large (77–80 kWh)"]
CLASS_COLOR = dict(zip(CLASSES, SERIES))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def load(name: str):
    for file in ("results.json", "baselines.json"):
        if (OUT / name / file).exists():
            return json.loads((OUT / name / file).read_text())
    return None


def vehicle_classes(data_dir: Path) -> dict[str, str]:
    caps = pd.read_csv(data_dir / "ev.csv", usecols=["vehicle_id", "maximumBatteryCapacity"]).drop_duplicates("vehicle_id")
    return {str(v): MODEL_TO_CLASS[CAPACITY_TO_MODEL[int(c)]] for v, c in zip(caps.vehicle_id, caps.maximumBatteryCapacity)}


def save(fig, name: str) -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", MEDIA / name)


def median_by_round(run: dict, key: str = "r2") -> tuple[list, list]:
    rounds, values = [], []
    for r in run["rounds"]:
        if r["clients"]:
            rounds.append(r["round"])
            values.append(np.nanmedian([c[key] for c in r["clients"]]))
    return rounds, values


def fig_non_iid(data_dir: Path, classes: dict) -> None:
    """Each client is a dot: how fast it drives vs how much energy it needs per km, and how much data it has."""
    series = load_ev(str(data_dir / "ev.csv"))
    rows = []
    for vid, steps in series.items():
        metres = steps[:, 0].sum()
        if metres < 500:
            continue
        moving = steps[steps[:, 0] > 0.5, 0]
        rows.append({"cls": classes[vid], "speed": moving.mean() * 3.6, "wh_km": steps[:, 2].sum() / (metres / 1000),
                     "windows": len(ev_windows(steps, EVTask())[0])})
    df = pd.DataFrame(rows)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for cls in CLASSES:
        d = df[df.cls == cls]
        ax.scatter(d.speed, d.wh_km, s=14, color=CLASS_COLOR[cls], alpha=0.75, edgecolors=SURFACE, linewidths=0.4,
                   label=f"{cls}  ·  {len(d)} cars")
    ax.set_ylim(df.wh_km.quantile(0.01) * 0.9, df.wh_km.quantile(0.99) * 1.1)
    ax.set_xlabel("mean driving speed (km/h)")
    ax.set_ylabel("energy per km (Wh/km)")
    ax.set_title("Feature and concept skew: each dot is one client (EV)")
    ax.legend(loc="upper left", markerscale=1.6)

    bins = np.linspace(0, df.windows.quantile(0.99), 30)
    for cls in CLASSES:
        ax2.hist(df[df.cls == cls].windows, bins=bins, histtype="step", linewidth=2, color=CLASS_COLOR[cls], label=cls)
    ax2.set_xlabel("training windows per client")
    ax2.set_ylabel("number of clients")
    ax2.set_title("Quantity skew: short and long trips")
    ax2.annotate(f"median {int(df.windows.median())} windows,\nlargest {int(df.windows.max()):,}",
                 xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", color=INK_2, fontsize=9)
    fig.suptitle("The simulated fleet is strongly non-IID", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "non_iid.png")


def fig_convergence() -> None:
    base = load("ev50-baselines")
    runs = [("ev50-fedavg", "r2", "FedAvg, 2 local epochs"), ("ev50-fedadam", "r2", "FedAdam, 2 local epochs"),
            ("ev50-fedavg-e5-ft", "r2", "FedAvg, 5 local epochs"), ("ev50-fedavg-e5-ft", "r2-ft", "FedAvg 5 ep + local fine-tune")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ends = []
    for (name, key, label), color in zip(runs, SERIES):
        run = load(name)
        if not run:
            continue
        x, y = median_by_round(run, key)
        ax.plot(x, y, color=color, label=label)
        ends.append((y[-1], x[-1]))
    # End labels, nudged apart vertically so close values don't overlap.
    placed = []
    for value, last_round in sorted(ends):
        pos = max([value] + [q + 0.07 for q in placed if abs(q - value) < 0.07])
        placed.append(pos)
        ax.annotate(f"{value:.2f}", (last_round, value), xytext=(last_round + 0.6, pos), textcoords="data",
                    va="center", color=INK_2, fontsize=9)
    if base:
        for key, label in (("centralized", "centralized (data pooled)"), ("local", "local only")):
            v = np.nanmedian([m["r2"] for m in base[key].values()])
            ax.axhline(v, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.annotate(f"{label}  {v:.2f}", (1, v), xytext=(2, 4), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("federated round")
    ax.set_ylabel("median R² over 50 clients (held-out)")
    ax.set_title("Global FedAvg trails local training; fine-tuning the global model locally beats it")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "convergence.png")


def fig_head_to_head(classes: dict) -> None:
    base, run = load("ev50-baselines"), load("ev50-fedavg-e5-ft")
    if not (base and run):
        return
    last = [r for r in run["rounds"] if r["clients"]][-1]["clients"]
    pts = [(base["local"][c["client"]]["mae"], c["mae-ft"], classes[c["client"]]) for c in last if c["client"] in base["local"]]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lo, hi = min(min(a, b) for a, b, _ in pts) * 0.8, max(max(a, b) for a, b, _ in pts) * 1.25
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=SERIES[2], alpha=0.06, linewidth=0)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    for cls in CLASSES:
        p = [(a, b) for a, b, c in pts if c == cls]
        ax.scatter(*zip(*p), s=46, color=CLASS_COLOR[cls], edgecolors=SURFACE, linewidths=1.2, label=cls, zorder=3)
    wins = np.mean([b < a for a, b, _ in pts])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(0.96, 0.06, f"below the line: FL wins\n{wins:.0%} of {len(pts)} clients", transform=ax.transAxes,
            ha="right", va="bottom", color=INK_2, fontsize=9)
    ax.set_xlabel("local-only model: MAE per 30 s (Wh)")
    ax.set_ylabel("FedAvg + local fine-tune: MAE (Wh)")
    ax.set_title("Per client: personalized FL vs training alone")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "head_to_head.png")


def fig_privacy() -> None:
    ref = load("ev50-dp0.0")
    pts = []
    for sigma in ("0.5", "1.0", "2.0"):
        run = load(f"ev50-dp{sigma}")
        if not run:
            continue
        clients = [r for r in run["rounds"] if r["clients"]][-1]["clients"]
        eps = np.array([c["epsilon"] for c in clients if c.get("epsilon", 0) > 0])
        r2 = np.array([c["r2"] for c in clients])
        pts.append((sigma, np.median(eps), np.percentile(eps, [25, 75]), np.nanmedian(r2), np.nanpercentile(r2, [25, 75])))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for sigma, e, e_iqr, r, r_iqr in pts:
        ax.errorbar(e, r, xerr=[[e - e_iqr[0]], [e_iqr[1] - e]], yerr=[[r - r_iqr[0]], [r_iqr[1] - r]],
                    fmt="o", color=SERIES[0], markersize=8, elinewidth=1.4, capsize=0, markeredgecolor=SURFACE)
        ax.annotate(f"σ = {sigma}", (e, r), xytext=(8, 8), textcoords="offset points", color=INK, fontsize=9)
    ax.plot([p[1] for p in pts], [p[3] for p in pts], color=SERIES[0], linewidth=1.2, alpha=0.5)
    if ref:
        v = np.nanmedian([c["r2"] for c in [r for r in ref["rounds"] if r["clients"]][-1]["clients"]])
        ax.axhline(v, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"no DP  {v:.2f}", (1, v), xytext=(2, 4), textcoords="offset points", color=INK_2, fontsize=9)
    ax.axvspan(0.05, 0.5, color=SERIES[1], alpha=0.08, linewidth=0)
    ax.text(0.07, -0.95, "v1 reported ε ≈ 0 at σ = 1.5–2.0\n(sampling rate 1/N, accountant reset per round)",
            color=INK_2, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 200)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("privacy budget ε after 20 rounds (log scale; lower = more private), δ = 1e-4")
    ax.set_ylabel("median R² over 50 clients")
    ax.set_title("Privacy–utility trade-off with correct accounting (median and interquartile range)")
    fig.tight_layout()
    save(fig, "privacy.png")


def fig_dropout() -> None:
    ref = load("ev50-fedavg-e5")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, d in zip(axes, ("0.3", "0.6")):
        if ref:
            x, y = median_by_round(ref)
            ax.plot(x, y, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), label="FedAvg, no dropout")
        for (strategy, label), color in zip((("fedavg", "FedAvg"), ("ftavg", "FaultTolerantFedAvg (needs 60%)")), SERIES):
            run = load(f"ev50-dropout{d}-{strategy}")
            if not run:
                continue
            x, y = median_by_round(run)
            ax.plot(x, y, color=color, label=label)
            rejected = [r["round"] for r in run["rounds"] if not r["train"]]
            if rejected:
                ax.scatter(rejected, [-0.95] * len(rejected), marker="|", s=60, color=color, linewidths=1.6)
                ax.annotate(f"{len(rejected)} rounds rejected (ticks)", (rejected[0], -0.95), xytext=(0, 8),
                            textcoords="offset points", color=INK_2, fontsize=8.5)
        ax.set_title(f"{int(float(d) * 100)}% of sampled clients drop out each round")
        ax.set_xlabel("federated round")
        ax.set_ylim(-1, 1)
    axes[0].set_ylabel("median R² over 50 clients")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "dropout.png")


BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential ramp


def fig_cs_demand(cs_dir: Path) -> None:
    """Energy delivered per station and 15-minute slot over the two simulated days."""
    if not (cs_dir / "cs.csv").exists():
        return
    from matplotlib.colors import LinearSegmentedColormap

    from evfl.data import load_cs

    series = load_cs(str(cs_dir / "cs.csv"), 900)
    names = sorted(series, key=lambda n: (n.split("_cs_")[0], int(n.split("_cs_")[1])))
    grid = np.stack([series[n] for n in names]) / 1000  # kWh per 15 min
    hours = np.arange(grid.shape[1]) * 0.25

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest",
                   extent=[0, hours[-1] + 0.25, len(names), 0], vmin=0, vmax=np.percentile(grid[grid > 0], 98))
    districts = [n.split("_cs_")[0] for n in names]
    starts = [i for i, d in enumerate(districts) if i == 0 or d != districts[i - 1]]
    for i in starts[1:]:
        ax.axhline(i, color=SURFACE, linewidth=2)
    ax.set_yticks([(s + e) / 2 for s, e in zip(starts, starts[1:] + [len(names)])])
    ax.set_yticklabels([f"{districts[s]} ({e - s})" for s, e in zip(starts, starts[1:] + [len(names)])])
    ax.grid(False)
    ax.set_title("Charging demand per station (district, number of stations) — the clients of use case (b)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("kWh per 15 min", color=INK_2)

    total = grid.sum(axis=0)
    ax2.fill_between(hours, total, step="post", color=BLUES[4], alpha=0.85, linewidth=0)
    ax2.set_ylabel("all stations\n(kWh / 15 min)")
    ax2.set_xlabel("simulated time (hours)")
    ax2.set_xticks(range(0, 49, 6))
    for day in (24, 48):
        ax2.axvline(day, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    fig.tight_layout()
    save(fig, "cs_demand.png")


def fig_cs_per_station() -> None:
    base, run = load("cs-baselines"), load("cs-fedavg-ft")
    if not (base and run):
        return
    last = {c["client"]: c for c in [r for r in run["rounds"] if r["clients"]][-1]["clients"]}
    names = sorted(set(last) & set(base["local"]), key=lambda n: base["local"][n]["mae"])
    rows = [("local only", [base["local"][n]["mae"] for n in names]),
            ("centralized (pooled)", [base["centralized"][n]["mae"] for n in names]),
            ("FedAvg + local fine-tune", [last[n]["mae-ft"] for n in names])]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(names))
    for (label, values), color in zip(rows, SERIES):
        ax.scatter(x, values, s=34, color=color, edgecolors=SURFACE, linewidths=1, label=f"{label}  ·  median {np.median(values):.0f} Wh", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("MAE per 15-min slot (Wh), held-out")
    ax.set_title("Use case (b): forecast error per charging station (sorted by local-only error)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "cs_per_station.png")


def main() -> None:
    data_dir = Path("data")
    classes = vehicle_classes(data_dir)
    fig_non_iid(data_dir, classes)
    fig_convergence()
    fig_head_to_head(classes)
    fig_privacy()
    fig_dropout()
    fig_cs_demand(data_dir / "cs")
    fig_cs_per_station()


if __name__ == "__main__":
    main()
