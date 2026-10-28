"""Regenerate the simulated EV and charging-station datasets with SUMO.

Pipeline: OSM road network (Overpass) -> netconvert -> charging stations on random
lanes -> randomTrips/duarouter -> SUMO run -> per-step CSVs for both use cases.

    python scenario/build.py                     # thesis-sized scenario (~1000 EVs)
    python scenario/build.py --end 20000 --cs 20 # smaller, faster run

Everything is written to --out (default: data/), and every step is skipped when
its output already exists, so a re-run only redoes what is missing.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Recovered from the OSM way IDs in the original v1 route file (~20 x 20 km around Stuttgart).
DEFAULT_BBOX = (48.79, 9.05, 48.97, 9.34)  # south, west, north, east

OVERPASS_SERVERS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

# Labels the thesis used for charging-station regions; stations get the nearest one.
DISTRICTS = {
    "Mitte": (48.7758, 9.1829),
    "Lud": (48.8974, 9.1916),
    "Esl": (48.7406, 9.3108),
    "Sind": (48.7133, 9.0028),
    "Renn": (48.7656, 8.9346),
}


def sumo_tools() -> Path:
    sumo = shutil.which("sumo")
    if sumo is None:
        sys.exit("SUMO not found on PATH (brew install sumo, or see eclipse.dev/sumo)")
    home = Path(sumo).resolve().parent.parent / "share" / "sumo"
    if not (home / "tools" / "randomTrips.py").exists():
        sys.exit(f"SUMO tools not found at {home / 'tools'}")
    # A stale SUMO_HOME (e.g. left over from an older install) breaks netconvert's typemaps.
    os.environ["SUMO_HOME"] = str(home)
    return home / "tools"


def run(cmd: list) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def fetch_osm(bbox: tuple, path: Path) -> None:
    """Download drivable roads only; the full OSM extract for this area is several hundred MB."""
    s, w, n, e = bbox
    query = f"""[out:xml][timeout:300];
        (way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|_link"]({s},{w},{n},{e}););
        (._;>;);
        out body;"""
    body = urllib.parse.urlencode({"data": query}).encode()
    tmp = path.with_suffix(".part")
    # Public Overpass servers are often busy; retry and fall back to a mirror.
    for attempt, server in enumerate(OVERPASS_SERVERS * 3):
        req = urllib.request.Request(server, body, headers={"User-Agent": "fl-thesis-scenario/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.rename(path)
            return
        except (urllib.error.URLError, TimeoutError) as err:
            print(f"  {server}: {err}; retrying", flush=True)
            time.sleep(10 * (attempt + 1))
    sys.exit("Could not download the OSM road network; try again later")


def place_charging_stations(net_path: Path, out_path: Path, count: int, seed: int, tools: Path) -> None:
    """Install `count` 22 kW stations on random passenger lanes, as the thesis did with netedit."""
    sys.path.insert(0, str(tools))
    import sumolib  # shipped with SUMO, not on PyPI in a matching version

    net = sumolib.net.readNet(str(net_path))
    lanes = [
        lane
        for edge in net.getEdges()
        if edge.allows("passenger") and edge.getFunction() == ""
        for lane in edge.getLanes()
        if lane.allows("passenger") and lane.getLength() > 30
    ]
    rng = random.Random(seed)
    root = ET.Element("additional")
    per_district = {d: 0 for d in DISTRICTS}
    for lane in rng.sample(lanes, count):
        x, y = lane.getShape()[0]
        lon, lat = net.convertXY2LonLat(x, y)
        district = min(DISTRICTS, key=lambda d: (DISTRICTS[d][0] - lat) ** 2 + (DISTRICTS[d][1] - lon) ** 2)
        cs_id = f"{district}_cs_{per_district[district]}"
        per_district[district] += 1
        start = rng.uniform(5, lane.getLength() - 15)
        ET.SubElement(root, "chargingStation", {
            "id": cs_id, "lane": lane.getID(),
            "startPos": f"{start:.2f}", "endPos": f"{start + 10:.2f}",
            "friendlyPos": "true", "power": "22000", "efficiency": "0.95",
            "chargeInTransit": "1", "chargeDelay": "0",
        })
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def add_charging_stops(trips_path: Path, stations_path: Path, fraction: float, duration: tuple, seed: int) -> None:
    """Send a share of trips via a random charging station, where they stop to charge.

    Without this, EVs only charge in transit while driving over a 10 m station
    (about one second per pass), which leaves the stations with almost no demand.
    """
    stations = [(cs.get("id"), cs.get("lane").rsplit("_", 1)[0])
                for cs in ET.parse(stations_path).getroot().iter("chargingStation")]
    rng = random.Random(seed)
    tree = ET.parse(trips_path)
    for trip in tree.getroot().iter("trip"):
        if rng.random() < fraction:
            cs_id, edge = rng.choice(stations)
            trip.set("via", edge)
            ET.SubElement(trip, "stop", {"chargingStation": cs_id, "duration": str(rng.randint(*duration))})
    tree.write(trips_path, encoding="utf-8", xml_declaration=True)


def battery_to_csv(xml_path: Path, csv_path: Path) -> int:
    """Flatten SUMO battery-output (one row per vehicle per step) for use case (a)."""
    fields = ["time", "vehicle_id", "speed", "acceleration", "energyConsumed",
              "totalEnergyConsumed", "totalEnergyRegenerated", "actualBatteryCapacity",
              "maximumBatteryCapacity", "chargingStationId", "energyCharged"]
    rows = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        time = None
        for event, el in ET.iterparse(xml_path, events=("start", "end")):
            if event == "start" and el.tag == "timestep":
                time = el.get("time")
            elif event == "end" and el.tag == "vehicle":
                writer.writerow([time, el.get("id")] + [el.get(k) for k in fields[2:]])
                rows += 1
                el.clear()
            elif event == "end" and el.tag == "timestep":
                el.clear()
    return rows


def charging_to_csv(xml_path: Path, csv_path: Path) -> int:
    """Flatten SUMO chargingstations-output (one row per charging step) for use case (b).

    Uses `energyCharged` (Wh in this step). `partialCharge` is the running total of the
    session, so summing it (as v1 did) overcounts the delivered energy.
    """
    rows = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chargingStation_id", "vehicle_id", "vehicle_type", "time", "energyCharged"])
        for station in ET.parse(xml_path).getroot().iter("chargingStation"):
            for vehicle in station.iter("vehicle"):
                for step in vehicle.iter("step"):
                    writer.writerow([station.get("id"), vehicle.get("id"), vehicle.get("type"),
                                     step.get("time"), step.get("energyCharged")])
                    rows += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("S", "W", "N", "E"))
    p.add_argument("--end", type=int, default=101000, help="last departure time in s (thesis: 101000)")
    p.add_argument("--period", type=float, default=100, help="s between departures (thesis: 100)")
    p.add_argument("--cs", type=int, default=50, help="number of charging stations")
    p.add_argument("--charge-fraction", type=float, default=0.0,
                   help="share of trips that detour to a charging station and stop there")
    p.add_argument("--stop-min", type=int, default=600, help="shortest charging stop in s")
    p.add_argument("--stop-max", type=int, default=2700, help="longest charging stop in s")
    p.add_argument("--no-battery-output", dest="battery_output", action="store_false",
                   help="skip the per-step battery log (large); enough for the charging-station use case")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tools = sumo_tools()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    osm, net = out / "map.osm.xml", out / "map.net.xml"
    stations, trips, routes = out / "charging_stations.add.xml", out / "trips.xml", out / "routes.rou.xml"
    battery_xml, charging_xml = out / "battery.out.xml", out / "charging.out.xml"

    if not osm.exists():
        print(f"Downloading road network for bbox {args.bbox} ...", flush=True)
        fetch_osm(tuple(args.bbox), osm)
    if not net.exists():
        # Same netconvert options as the thesis (ch. 5.1).
        run(["netconvert", "--osm-files", osm, "--output-file", net,
             "--geometry.remove", "--roundabouts.guess", "--ramps.guess", "--junctions.join",
             "--tls.guess-signals", "--tls.discard-simple", "--tls.join", "--no-warnings"])
    if not stations.exists():
        place_charging_stations(net, stations, args.cs, args.seed, tools)
    if not routes.exists():
        trip_args = ["-n", net, "-o", trips, "-e", args.end, "--period", args.period, "--fringe-factor", 1,
                     "--seed", args.seed, "--additional-file", HERE / "vehicles.add.xml",
                     "--trip-attributes", 'type="typedist1"']
        if args.charge_fraction == 0:
            run([sys.executable, tools / "randomTrips.py", *trip_args, "-r", routes, "--validate"])
        else:
            run([sys.executable, tools / "randomTrips.py", *trip_args])
            add_charging_stops(trips, stations, args.charge_fraction, (args.stop_min, args.stop_max), args.seed)
            run(["duarouter", "-n", net, "--route-files", trips, "-o", routes,
                 "-a", f"{HERE / 'vehicles.add.xml'},{stations}", "--ignore-errors", "--no-warnings"])

    done_marker = battery_xml if args.battery_output else charging_xml
    if not done_marker.exists():
        # The route file already carries the vehicle types (randomTrips/duarouter copy them in).
        outputs = ["--battery-output", battery_xml, "--battery-output.precision", 4] if args.battery_output else []
        try:
            run(["sumo", "-n", net, "-r", routes, "-a", stations, *outputs,
                 "--chargingstations-output", charging_xml,
                 "--device.battery.probability", 1, "--seed", args.seed,
                 "--no-step-log", "--no-warnings", "--duration-log.statistics"])
        except subprocess.CalledProcessError:
            # SUMO leaves truncated outputs behind; drop them so a re-run retries this step.
            battery_xml.unlink(missing_ok=True)
            charging_xml.unlink(missing_ok=True)
            raise

    if args.battery_output:
        print(f"EV rows: {battery_to_csv(battery_xml, out / 'ev.csv'):,}")
    print(f"CS rows: {charging_to_csv(charging_xml, out / 'cs.csv'):,}")


if __name__ == "__main__":
    main()
