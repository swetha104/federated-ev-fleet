"""Render an animation of the simulated fleet for the docs (no GUI or XQuartz needed).

Re-runs a time window of a built scenario headless, records each car's
position and battery level every few seconds, and draws them over the road
network: dots are cars coloured by battery class (small/mid/large, the same
grouping as the README figures), ringed in white while charging; triangles are
the charging stations.

    python scenario/render.py --data-dir data/cs --out docs/media/fleet.gif
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


def road_segments(net_path: Path) -> list:
    """First-lane shape of every non-internal edge, as line segments in SUMO's x/y metres."""
    segments = []
    for _, el in ET.iterparse(net_path, events=("end",)):
        if el.tag == "edge":
            if el.get("function") != "internal":
                lane = el.find("lane")
                if lane is not None:
                    segments.append([tuple(map(float, p.split(","))) for p in lane.get("shape").split()])
            el.clear()
    return segments


def station_positions(stations_path: Path, net_path: Path) -> list:
    lanes = {}
    for _, el in ET.iterparse(net_path, events=("end",)):
        if el.tag == "lane":
            lanes[el.get("id")] = el.get("shape").split()[0]
        elif el.tag == "edge":
            el.clear()
    return [tuple(map(float, lanes[cs.get("lane")].split(",")))
            for cs in ET.parse(stations_path).getroot().iter("chargingStation") if cs.get("lane") in lanes]


# The 7 EV models in scenario/vehicles.add.xml, grouped by battery size into three
# classes; dark-surface steps of the same palette as scripts/figures.py.
CLASS_COLORS = {"small (32–39 kWh)": "#3987e5", "mid (58–64 kWh)": "#d95926", "large (77–80 kWh)": "#199e70"}
TYPE_CLASS = {
    "VW_eUp": "small (32–39 kWh)", "BMW_i3": "small (32–39 kWh)",
    "VW_ID3": "mid (58–64 kWh)", "soulEV65": "mid (58–64 kWh)",
    "VW_ID4": "large (77–80 kWh)", "Kia_EV6_GT": "large (77–80 kWh)", "SUV": "large (77–80 kWh)",
}


def record(data_dir: Path, begin: int, end: int, period: int) -> dict:
    """time -> list of (x, y, vehicle type, charging?) from a headless SUMO run."""
    sumo_home = Path(shutil.which("sumo")).resolve().parent.parent / "share" / "sumo"
    env = {**os.environ, "SUMO_HOME": str(sumo_home)}
    with tempfile.TemporaryDirectory() as tmp:
        fcd = Path(tmp) / "fcd.xml"
        subprocess.run([
            "sumo", "-n", data_dir / "map.net.xml", "-r", data_dir / "routes.rou.xml",
            "-a", data_dir / "charging_stations.add.xml", "--device.battery.probability", "1",
            "--end", str(end), "--fcd-output", fcd, "--device.fcd.begin", str(begin), "--device.fcd.period", str(period),
            "--fcd-output.params",
            "device.battery.chargingStationId",
            "--no-step-log", "--no-warnings",
        ], check=True, env=env)
        frames = defaultdict(list)
        time = None
        for event, el in ET.iterparse(fcd, events=("start", "end")):
            if event == "start" and el.tag == "timestep":
                time = float(el.get("time"))
            elif event == "end" and el.tag == "vehicle":
                frames[time].append((float(el.get("x")), float(el.get("y")), el.get("type"),
                                     el.get("device.battery.chargingStationId") != "NULL"))
                el.clear()
        return dict(sorted(frames.items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/cs"))
    p.add_argument("--out", type=Path, default=Path("docs/media/fleet.gif"))
    p.add_argument("--begin", type=int, default=7 * 3600, help="start of the window in simulated seconds")
    p.add_argument("--end", type=int, default=9 * 3600)
    p.add_argument("--period", type=int, default=30, help="simulated seconds per frame")
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    if shutil.which("sumo") is None:
        sys.exit("SUMO not found on PATH")
    print("Reading road network ...", flush=True)
    roads = road_segments(args.data_dir / "map.net.xml")
    stations = station_positions(args.data_dir / "charging_stations.add.xml", args.data_dir / "map.net.xml")
    print(f"Simulating {args.begin}-{args.end} s ...", flush=True)
    frames = record(args.data_dir, args.begin, args.end, args.period)
    times = list(frames)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
    fig.patch.set_facecolor("#111418")
    ax.set_facecolor("#111418")
    ax.add_collection(LineCollection(roads, colors="#3a4048", linewidths=0.3))
    ax.scatter(*zip(*stations), marker="^", s=45, c="#9aa5b1", edgecolors="none", zorder=3, label="charging station")
    for name, color in CLASS_COLORS.items():
        ax.scatter([], [], s=20, c=color, label=name)
    ax.scatter([], [], s=40, facecolors="none", edgecolors="#ffffff", linewidths=1.2, label="charging now")
    cars = ax.scatter([], [], s=16, zorder=4)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    title = ax.set_title("", color="#e6e6e6", fontsize=11, loc="left")
    ax.legend(loc="lower right", facecolor="#1d2228", edgecolor="none", labelcolor="#e6e6e6", fontsize=8)
    fig.tight_layout()

    def draw(i):
        rows = frames[times[i]]
        cars.set_offsets([(x, y) for x, y, _, _ in rows] or [(float("nan"), float("nan"))])
        cars.set_facecolors([CLASS_COLORS[TYPE_CLASS[t]] if t in TYPE_CLASS else "#cccccc" for _, _, t, _ in rows])
        cars.set_edgecolors(["#ffffff" if charging else "none" for *_, charging in rows])
        cars.set_linewidths([1.2 if charging else 0 for *_, charging in rows])
        cars.set_sizes([40 if charging else 16 for *_, charging in rows])
        charging = sum(c for *_, c in rows)
        clock = int(times[i])
        title.set_text(f"Simulated Stuttgart fleet  {clock // 3600 % 24:02d}:{clock % 3600 // 60:02d}  "
                       f"·  {len(rows)} EVs on the road  ·  {charging} charging")
        return cars, title

    args.out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=len(times), blit=False).save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"Wrote {args.out} ({len(times)} frames, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
