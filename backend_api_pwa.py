from __future__ import annotations

import os
import math
import time
import json
import hashlib
import zipfile
import tempfile
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, date

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# ============================================================
# BusMaps Hanoi — Next Upgrade
# - Correct GTFS model (trip/pattern, directed)
# - Time-dependent transit routing (simplified RAPTOR)
# - Fast spatial index (grid) for nearest stops & transfers
# - OSRM-based polylines with waypoint compression + cache
# ============================================================

APP_TITLE = "BusMaps Hà Nội"
DEFAULT_PORT = int(os.getenv("PORT", "5000"))

# Walk and transfer parameters (tweak via env)
WALK_SPEED_MPS = float(os.getenv("WALK_SPEED_MPS", "1.4"))           # ~5 km/h
WALK_RADIUS_M = int(os.getenv("WALK_RADIUS_M", "800"))               # start/end to stops
TRANSFER_RADIUS_M = int(os.getenv("TRANSFER_RADIUS_M", "180"))       # between stops (walking transfer)
MAX_TRANSFERS = int(os.getenv("MAX_TRANSFERS", "4"))                 # number of bus legs - 1; RAPTOR rounds
MAX_CANDIDATE_STOPS = int(os.getenv("MAX_CANDIDATE_STOPS", "18"))    # start/end candidates

# OSRM
OSRM_BASE = os.getenv("OSRM_BASE", "https://router.project-osrm.org")
OSRM_TIMEOUT_S = float(os.getenv("OSRM_TIMEOUT_S", "8.0"))
OSRM_SNAP_RADIUS_M = int(os.getenv("OSRM_SNAP_RADIUS_M", "60"))
OSRM_WAYPOINT_SPACING_M = int(os.getenv("OSRM_WAYPOINT_SPACING_M", "250"))
OSRM_MAX_WAYPOINTS = int(os.getenv("OSRM_MAX_WAYPOINTS", "28"))

# Caching
CACHE_DIR = os.getenv("BUSMAPS_CACHE_DIR", os.path.join(os.getcwd(), ".busmaps_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
OSRM_CACHE_JSON = os.path.join(CACHE_DIR, "osrm_cache.json")

EARTH_R = 6371000.0

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two points."""
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * (math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_R * c

def fmt_hhmm(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    hh = (seconds // 3600) % 48  # allow >24h
    mm = (seconds % 3600) // 60
    return f"{hh:02d}:{mm:02d}"

def parse_hhmm_to_sec(s: str) -> Optional[int]:
    """Parse time string to seconds. Accepts 'HH:MM', 'HH:MM:SS', and optional AM/PM suffix."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # Normalize common AM/PM formats
    # Examples: '09:16 AM', '9:16PM', '09:16:00 am'
    ampm = None
    parts_ws = s.split()
    if len(parts_ws) >= 2:
        # last token might be AM/PM
        tail = parts_ws[-1].upper()
        if tail in ('AM', 'PM'):
            ampm = tail
            s = ' '.join(parts_ws[:-1]).strip()
    else:
        tail = s[-2:].upper()
        if tail in ('AM', 'PM'):
            ampm = tail
            s = s[:-2].strip()

    # Accept 'HH:MM' or 'HH:MM:SS'
    parts = s.split(':')
    if len(parts) < 2:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2]) if len(parts) >= 3 else 0
        if ampm:
            hh = hh % 12
            if ampm == 'PM':
                hh += 12
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return None

def now_hhmm_local() -> str:
    # Use local time; server runs on user's machine typically.
    return datetime.now().strftime("%H:%M")

# -------------------- OSRM cache --------------------
def _load_osrm_cache() -> Dict[str, Any]:
    try:
        with open(OSRM_CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_osrm_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(OSRM_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass

_OSRM_CACHE = _load_osrm_cache()

def _osrm_cache_key(profile: str, coords: List[Tuple[float, float]], radiuses: int) -> str:
    # coords: list of (lat, lon)
    h = hashlib.sha1()
    h.update(profile.encode("utf-8"))
    h.update(str(radiuses).encode("utf-8"))
    for lat, lon in coords:
        h.update(f"{lat:.6f},{lon:.6f};".encode("utf-8"))
    return h.hexdigest()

def compress_waypoints(coords: List[Tuple[float, float]],
                       spacing_m: int = OSRM_WAYPOINT_SPACING_M,
                       max_points: int = OSRM_MAX_WAYPOINTS) -> List[Tuple[float, float]]:
    """Keep first/last, and sample intermediate points by distance."""
    if len(coords) <= 2:
        return coords
    keep: List[Tuple[float, float]] = [coords[0]]
    last = coords[0]
    for c in coords[1:-1]:
        if haversine_m(last[0], last[1], c[0], c[1]) >= spacing_m:
            keep.append(c)
            last = c
            if len(keep) >= max_points - 1:
                break
    keep.append(coords[-1])
    return keep

def osrm_route(profile: str, coords_latlon: List[Tuple[float, float]],
               snap_radius_m: int = OSRM_SNAP_RADIUS_M) -> Optional[Dict[str, Any]]:
    """
    Return dict: {path:[[lat,lon],...], distance_m, duration_s}
    """
    if len(coords_latlon) < 2:
        return None
    try:
        coords_latlon = compress_waypoints(coords_latlon)
        key = _osrm_cache_key(profile, coords_latlon, snap_radius_m)
        cached = _OSRM_CACHE.get(key)
    except Exception:
        # Never crash routing because of cache/key issues
        key = None
        cached = None
    if isinstance(cached, dict) and "path" in cached:
        return cached

    # Build OSRM URL (lon,lat;lon,lat...)
    coords_str = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in coords_latlon])
    radiuses = ";".join([str(int(snap_radius_m))] * len(coords_latlon))
    url = f"{OSRM_BASE.rstrip('/')}/route/v1/{profile}/{coords_str}?overview=full&geometries=geojson&steps=false&radiuses={radiuses}"
    try:
        import requests
        r = requests.get(url, timeout=OSRM_TIMEOUT_S)
        if r.status_code != 200:
            return None
        j = r.json()
        routes = j.get("routes") or []
        if not routes:
            return None
        route0 = routes[0]
        geom = route0.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if not coords:
            return None
        path = [[c[1], c[0]] for c in coords]  # to [lat,lon]
        out = {
            "path": path,
            "distance_m": float(route0.get("distance") or 0.0),
            "duration_s": float(route0.get("duration") or 0.0),
        }
        if key:
            _OSRM_CACHE[key] = out
        # Save occasionally
        if len(_OSRM_CACHE) % 50 == 0:
            _save_osrm_cache(_OSRM_CACHE)
        return out
    except Exception:
        return None

# -------------------- Spatial index --------------------
class GridIndex:
    """
    Very fast stop candidate search using fixed-size grid.
    cell_deg ~ 0.01 => ~1.1km in lat.
    """
    def __init__(self, cell_deg: float = 0.01):
        self.cell_deg = cell_deg
        self.cells: Dict[Tuple[int, int], List[str]] = {}

    def _key(self, lat: float, lon: float) -> Tuple[int, int]:
        return (int(lat / self.cell_deg), int(lon / self.cell_deg))

    def add(self, stop_id: str, lat: float, lon: float):
        k = self._key(lat, lon)
        self.cells.setdefault(k, []).append(stop_id)

    def query_radius(self, lat: float, lon: float, radius_m: float,
                     stops_ll: Dict[str, Tuple[float, float]],
                     limit: int = 50) -> List[Tuple[str, float]]:
        # Determine how many cells to search
        # conservative conversion: 1 deg lat ~ 111km
        deg = radius_m / 111000.0
        step = max(1, int(math.ceil(deg / self.cell_deg)))
        ck = self._key(lat, lon)
        cand: List[Tuple[str, float]] = []
        for dy in range(-step, step + 1):
            for dx in range(-step, step + 1):
                ids = self.cells.get((ck[0] + dy, ck[1] + dx))
                if not ids:
                    continue
                for sid in ids:
                    slat, slon = stops_ll[sid]
                    d = haversine_m(lat, lon, slat, slon)
                    if d <= radius_m:
                        cand.append((sid, d))
        cand.sort(key=lambda x: x[1])
        return cand[:limit]

# -------------------- GTFS model --------------------
@dataclass
class Stop:
    stop_id: str
    stop_name: str
    lat: float
    lon: float

@dataclass
class Route:
    route_id: str
    short_name: str
    long_name: str

@dataclass
class Trip:
    trip_id: str
    route_id: str
    service_id: str

@dataclass
class PatternTripTimes:
    trip_id: str
    arr: List[int]
    dep: List[int]

@dataclass
class Pattern:
    pattern_id: int
    route_id: str
    stops: List[str]
    trips: List[PatternTripTimes]  # all trips of this pattern

# -------------------- Backend state --------------------
GTFS_OK = False
GTFS_MESSAGE = "Not loaded"
GTFS_PATH_USED = None

STOPS: Dict[str, Stop] = {}
STOP_LL: Dict[str, Tuple[float, float]] = {}
ROUTES: Dict[str, Route] = {}
TRIPS: Dict[str, Trip] = {}
PATTERNS: Dict[int, Pattern] = {}
# stop -> list of (pattern_id, stop_index)
STOP_TO_PATTERNS: Dict[str, List[Tuple[int, int]]] = {}
CONNECTED_STOP_IDS: Set[str] = set()  # stops served by at least one pattern
GRID = GridIndex(cell_deg=0.01)
# walking neighbors for transfers
STOP_NEIGHBORS: Dict[str, List[Tuple[str, float]]] = {}  # stop -> [(neighbor_stop, dist_m)]
# lazy board index cache: (pattern_id, stop_index) -> list of (dep_time, trip_idx)
BOARD_INDEX: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
ROUTE_EDGES: Dict[str, List[Tuple[str, str, int]]] = {}  # from_stop -> [(to_stop, route_id, pattern_id)]

def _read_gtfs_csv(folder: str, name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return None

def _try_find_gtfs_folder() -> Optional[str]:
    # Priority: env GTFS_PATH
    env = os.getenv("GTFS_PATH")
    if env and os.path.exists(env):
        return env

    # Look near current working dir and file dir
    candidates: List[str] = []
    cwd = os.getcwd()
    candidates.append(os.path.join(cwd, "hanoi_gtfs_md"))
    candidates.append(os.path.join(cwd, "gtfs"))
    candidates.append(cwd)

    # Scan for folders containing stops.txt and stop_times.txt
    for base in candidates:
        if not os.path.exists(base):
            continue
        if os.path.isdir(base):
            if os.path.exists(os.path.join(base, "stops.txt")) and os.path.exists(os.path.join(base, "stop_times.txt")):
                return base

    # Scan for a zip file in cwd
    for fn in os.listdir(cwd):
        if fn.lower().endswith(".zip") and "gtfs" in fn.lower():
            zpath = os.path.join(cwd, fn)
            try:
                tmpdir = os.path.join(CACHE_DIR, "gtfs_unzipped")
                if os.path.exists(tmpdir):
                    shutil.rmtree(tmpdir, ignore_errors=True)
                os.makedirs(tmpdir, exist_ok=True)
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(tmpdir)
                # some zips have nested folder
                # find any directory containing required files
                for root, dirs, files in os.walk(tmpdir):
                    if "stops.txt" in files and "stop_times.txt" in files and "trips.txt" in files and "routes.txt" in files:
                        return root
            except Exception:
                continue
    return None

def _build_board_index(pattern: Pattern, stop_index: int) -> List[Tuple[int, int]]:
    key = (pattern.pattern_id, stop_index)
    cached = BOARD_INDEX.get(key)
    if cached is not None:
        return cached
    pairs: List[Tuple[int, int]] = []
    for ti, t in enumerate(pattern.trips):
        if stop_index < len(t.dep):
            dep = t.dep[stop_index]
            if dep is not None:
                pairs.append((int(dep), ti))
    pairs.sort(key=lambda x: x[0])
    BOARD_INDEX[key] = pairs
    return pairs

def _find_earliest_trip(pattern: Pattern, stop_index: int, earliest_time: int) -> Optional[Tuple[PatternTripTimes, int]]:
    pairs = _build_board_index(pattern, stop_index)
    # binary search for first dep >= earliest_time
    lo, hi = 0, len(pairs)
    while lo < hi:
        mid = (lo + hi) // 2
        if pairs[mid][0] < earliest_time:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(pairs):
        return None
    dep, ti = pairs[lo]
    return (pattern.trips[ti], dep)

def _precompute_neighbors():
    # For each stop, find nearby stops within TRANSFER_RADIUS_M (limited)
    global STOP_NEIGHBORS
    STOP_NEIGHBORS = {}
    for sid, (lat, lon) in STOP_LL.items():
        cand = GRID.query_radius(lat, lon, TRANSFER_RADIUS_M, STOP_LL, limit=40)
        # remove self
        neigh = [(nid, d) for nid, d in cand if nid != sid]
        STOP_NEIGHBORS[sid] = neigh

def load_gtfs() -> None:
    global GTFS_OK, GTFS_MESSAGE, GTFS_PATH_USED
    global STOPS, STOP_LL, ROUTES, TRIPS, PATTERNS, STOP_TO_PATTERNS, CONNECTED_STOP_IDS, GRID, BOARD_INDEX, ROUTE_EDGES

    t0 = time.time()
    folder = _try_find_gtfs_folder()
    if not folder:
        GTFS_OK = False
        GTFS_MESSAGE = "Không tìm thấy GTFS. Hãy đặt biến môi trường GTFS_PATH trỏ tới thư mục chứa stops.txt/stop_times.txt."
        GTFS_PATH_USED = None
        return

    stops = _read_gtfs_csv(folder, "stops.txt")
    routes = _read_gtfs_csv(folder, "routes.txt")
    trips = _read_gtfs_csv(folder, "trips.txt")
    stop_times = _read_gtfs_csv(folder, "stop_times.txt")
    if stops is None or routes is None or trips is None or stop_times is None:
        GTFS_OK = False
        GTFS_MESSAGE = f"GTFS thiếu file bắt buộc trong {folder}. Cần: stops.txt, routes.txt, trips.txt, stop_times.txt."
        GTFS_PATH_USED = folder
        return

    # Build stops
    STOPS = {}
    STOP_LL = {}
    GRID = GridIndex(cell_deg=0.01)
    for _, r in stops.iterrows():
        sid = str(r.get("stop_id", "")).strip()
        if not sid:
            continue
        try:
            lat = float(r.get("stop_lat"))
            lon = float(r.get("stop_lon"))
        except Exception:
            continue
        name = str(r.get("stop_name", "") or "").strip()
        STOPS[sid] = Stop(stop_id=sid, stop_name=name, lat=lat, lon=lon)
        STOP_LL[sid] = (lat, lon)
        GRID.add(sid, lat, lon)

    # Build routes
    ROUTES = {}
    for _, r in routes.iterrows():
        rid = str(r.get("route_id", "")).strip()
        if not rid:
            continue
        ROUTES[rid] = Route(
            route_id=rid,
            short_name=str(r.get("route_short_name", "") or "").strip(),
            long_name=str(r.get("route_long_name", "") or "").strip(),
        )

    # Build trips
    TRIPS = {}
    for _, r in trips.iterrows():
        tid = str(r.get("trip_id", "")).strip()
        if not tid:
            continue
        rid = str(r.get("route_id", "")).strip()
        sid = str(r.get("service_id", "")).strip()
        TRIPS[tid] = Trip(trip_id=tid, route_id=rid, service_id=sid)

    # stop_times cleanup & parse
    # Keep required columns
    stop_times = stop_times[["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]].copy()
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce").fillna(0).astype(int)
    stop_times.sort_values(["trip_id", "stop_sequence"], inplace=True)

    # Group by trip_id (correct GTFS)
    trip_groups = stop_times.groupby("trip_id", sort=False)

    # Build patterns (group by route_id + stop sequence signature)
    PATTERNS = {}
    STOP_TO_PATTERNS = {}
    BOARD_INDEX = {}
    ROUTE_EDGES = {}
    pattern_map: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    pattern_id_seq = 1

    # Temporary: store pattern->list of trips times
    pattern_trips: Dict[int, List[PatternTripTimes]] = {}
    pattern_route: Dict[int, str] = {}
    pattern_stops: Dict[int, List[str]] = {}

    for trip_id, g in trip_groups:
        trip_id = str(trip_id)
        trip = TRIPS.get(trip_id)
        if not trip:
            continue
        stop_ids = [str(x) for x in g["stop_id"].tolist()]
        if len(stop_ids) < 2:
            continue

        # parse times aligned with stop_ids
        arr = []
        dep = []
        ok = True
        for a, d in zip(g["arrival_time"].tolist(), g["departure_time"].tolist()):
            ta = parse_hhmm_to_sec(str(a))
            td = parse_hhmm_to_sec(str(d))
            if ta is None or td is None:
                ok = False
                break
            arr.append(int(ta))
            dep.append(int(td))
        if not ok:
            continue

        key = (trip.route_id, tuple(stop_ids))
        pid = pattern_map.get(key)
        if pid is None:
            pid = pattern_id_seq
            pattern_id_seq += 1
            pattern_map[key] = pid
            pattern_route[pid] = trip.route_id
            pattern_stops[pid] = stop_ids
            pattern_trips[pid] = []

        pattern_trips[pid].append(PatternTripTimes(trip_id=trip_id, arr=arr, dep=dep))

    # Finalize patterns and indexes
    for pid, trips_list in pattern_trips.items():
        rid = pattern_route[pid]
        stops_list = pattern_stops[pid]
        # Sort trips by first departure (helps consistency)
        trips_list.sort(key=lambda t: t.dep[0] if t.dep else 10**12)
        PATTERNS[pid] = Pattern(pattern_id=pid, route_id=rid, stops=stops_list, trips=trips_list)
        # Directed connectivity edges (fallback routing)
        for j in range(len(stops_list) - 1):
            a = stops_list[j]
            b = stops_list[j + 1]
            ROUTE_EDGES.setdefault(a, []).append((b, rid, pid))


        for idx, sid in enumerate(stops_list):
            STOP_TO_PATTERNS.setdefault(sid, []).append((pid, idx))

        # Stops that are served by at least one pattern (used by connected_only filters)
    CONNECTED_STOP_IDS = set(STOP_TO_PATTERNS.keys())

# Precompute transfer neighbors
    _precompute_neighbors()

    GTFS_OK = True
    GTFS_PATH_USED = folder
    GTFS_MESSAGE = f"GTFS loaded from {folder}. stops={len(STOPS)}, routes={len(ROUTES)}, patterns={len(PATTERNS)}"
    dt = time.time() - t0
    GTFS_MESSAGE += f", load_time={dt:.2f}s"

# Load at startup
load_gtfs()

# -------------------- Routing (RAPTOR simplified) --------------------
INF = 10**12

def _fallback_topology_route(start_candidates: List[Tuple[str, float]],
                             end_stop_id: str) -> Optional[Dict[str, Any]]:
    """Fallback when time-dependent routing fails.

    Uses directed edges derived from patterns + walking transfer edges to find a feasible path.
    Produces itinerary without guaranteed schedule times (but still groups by route).
    """
    if not start_candidates or end_stop_id not in STOPS:
        return None

    # Choose best start stop (min walk)
    start_candidates = sorted(start_candidates, key=lambda x: x[1])
    start_stop = start_candidates[0][0]
    start_walk_m = float(start_candidates[0][1])

    from collections import deque
    q = deque([start_stop])
    prev_edge: Dict[str, Tuple[str, str, int, str]] = {}  # stop -> (prev_stop, mode, route_id, pattern_id_str)
    seen = {start_stop}

    while q:
        u = q.popleft()
        if u == end_stop_id:
            break

        # Bus edges
        for v, rid, pid in ROUTE_EDGES.get(u, []):
            if v not in seen:
                seen.add(v)
                prev_edge[v] = (u, "bus", rid, str(pid))
                q.append(v)

        # Walking transfers
        for v, d in STOP_NEIGHBORS.get(u, []):
            if v not in seen:
                seen.add(v)
                prev_edge[v] = (u, "walk", "", "")
                q.append(v)

    if end_stop_id not in seen:
        return None

    # Reconstruct stop path and edge modes
    path_stops = [end_stop_id]
    edge_modes: List[Tuple[str, str, str]] = []  # (mode, route_id, pattern_id)
    cur = end_stop_id
    while cur != start_stop:
        pu, mode, rid, pid = prev_edge[cur]
        edge_modes.append((mode, rid, pid))
        path_stops.append(pu)
        cur = pu
    path_stops.reverse()
    edge_modes.reverse()

    # Convert to segments grouped by consecutive mode/route
    segments: List[Dict[str, Any]] = []
    i = 0
    while i < len(edge_modes):
        mode, rid, pid = edge_modes[i]
        j = i
        while j + 1 < len(edge_modes) and edge_modes[j + 1][0] == mode and (mode != "bus" or edge_modes[j + 1][1] == rid):
            j += 1
        seg_stops = path_stops[i: j + 2]  # inclusive ends
        if mode == "bus":
            rinfo = ROUTES.get(rid)
            segments.append({
                "mode": "bus",
                "route_id": rid,
                "route_short_name": rinfo.short_name if rinfo else rid,
                "route_long_name": rinfo.long_name if rinfo else "",
                "from_stop_id": seg_stops[0],
                "to_stop_id": seg_stops[-1],
                "stop_ids": seg_stops,
                "pattern_id": pid,
                "board_time": None,
                "alight_time": None,
            })
        else:
            # walking transfer between two stops only (keep simple)
            segments.append({
                "mode": "walk",
                "from_stop_id": seg_stops[0],
                "to_stop_id": seg_stops[-1],
                "stop_ids": seg_stops,
            })
        i = j + 1

    return {
        "start_stop_id": start_stop,
        "start_walk_m": start_walk_m,
        "end_stop_id": end_stop_id,
        "segments": segments,
        "fallback": True,
    }



@dataclass
class BackPointer:
    mode: str  # 'walk_start', 'walk', 'bus'
    prev_stop: Optional[str] = None
    prev_coord: Optional[Tuple[float, float]] = None  # for walk_start
    walk_m: float = 0.0
    walk_s: float = 0.0
    pattern_id: Optional[int] = None
    trip_id: Optional[str] = None
    from_index: Optional[int] = None
    to_index: Optional[int] = None
    board_time: Optional[int] = None
    alight_time: Optional[int] = None

def _candidate_stops(lat: float, lon: float, radius_m: int, limit: int) -> List[Tuple[str, float]]:
    if not GTFS_OK:
        return []
    return GRID.query_radius(lat, lon, radius_m, STOP_LL, limit=limit)

def plan_route(start_lat: float, start_lon: float,
               end_lat: float, end_lon: float,
               end_stop_id: Optional[str],
               depart_time_hhmm: Optional[str],
               max_transfers: int = MAX_TRANSFERS) -> Tuple[bool, Any]:
    if not GTFS_OK:
        return False, {"error": "Backend chưa sẵn sàng (GTFS chưa load).", "message": GTFS_MESSAGE}

    if not depart_time_hhmm:
        depart_time_hhmm = now_hhmm_local()
    dep0 = parse_hhmm_to_sec(depart_time_hhmm)
    if dep0 is None:
        dep0 = parse_hhmm_to_sec(now_hhmm_local()) or 0
        depart_time_hhmm = fmt_hhmm(dep0)

    # End stop override if provided
    if end_stop_id and end_stop_id in STOPS:
        s = STOPS[end_stop_id]
        end_lat, end_lon = s.lat, s.lon

    start_cand = _candidate_stops(start_lat, start_lon, WALK_RADIUS_M, MAX_CANDIDATE_STOPS)
    end_cand = _candidate_stops(end_lat, end_lon, WALK_RADIUS_M, MAX_CANDIDATE_STOPS)

    if not start_cand:
        # Expand radius once as fallback
        start_cand = _candidate_stops(start_lat, start_lon, WALK_RADIUS_M * 2, MAX_CANDIDATE_STOPS)
    if not end_cand:
        end_cand = _candidate_stops(end_lat, end_lon, WALK_RADIUS_M * 2, MAX_CANDIDATE_STOPS)

    if not start_cand:
        return False, {"error": "Không tìm thấy trạm gần điểm xuất phát.", "suggestion": "Hãy chọn điểm xuất phát gần trạm xe buýt hơn."}
    if not end_cand:
        return False, {"error": "Không tìm thấy trạm gần điểm đến.", "suggestion": "Hãy chọn điểm đến gần trạm xe buýt hơn."}

    # earliest arrival time at stop
    earliest: Dict[str, int] = {sid: INF for sid in STOPS.keys()}
    prev: Dict[str, BackPointer] = {}

    # Initialize from start with walking
    for sid, dist in start_cand:
        t_walk = dist / WALK_SPEED_MPS
        t = int(dep0 + t_walk)
        if t < earliest[sid]:
            earliest[sid] = t
            prev[sid] = BackPointer(mode="walk_start", prev_coord=(start_lat, start_lon), walk_m=float(dist), walk_s=float(t_walk))

    # Helper: relax walking transfers
    def relax_walk_transfers(updated_stops: Set[str]) -> Set[str]:
        changed: Set[str] = set()
        for sid in list(updated_stops):
            t0a = earliest.get(sid, INF)
            if t0a >= INF:
                continue
            for nid, d in STOP_NEIGHBORS.get(sid, []):
                t = int(t0a + d / WALK_SPEED_MPS)
                if t < earliest[nid]:
                    earliest[nid] = t
                    prev[nid] = BackPointer(mode="walk", prev_stop=sid, walk_m=float(d), walk_s=float(d / WALK_SPEED_MPS))
                    changed.add(nid)
        return changed

    # Evaluate best end at any time
    def best_end() -> Tuple[int, str, float]:
        best_t = INF
        best_sid = ""
        best_walk = 0.0
        for sid, dist in end_cand:
            t = earliest.get(sid, INF)
            if t >= INF:
                continue
            total = int(t + dist / WALK_SPEED_MPS)
            if total < best_t:
                best_t = total
                best_sid = sid
                best_walk = float(dist)
        return best_t, best_sid, best_walk

    # Initial walking transfers (so boarding can use nearby stops)
    changed = relax_walk_transfers(set([sid for sid, _ in start_cand]))

    # RAPTOR rounds
    for _round in range(max_transfers + 1):
        marked = {sid for sid in STOPS.keys() if earliest[sid] < INF}
        # collect affected patterns
        patterns_to_scan: Set[int] = set()
        for sid in marked:
            for pid, _ in STOP_TO_PATTERNS.get(sid, []):
                patterns_to_scan.add(pid)

        any_update: Set[str] = set()
        for pid in patterns_to_scan:
            pat = PATTERNS.get(pid)
            if not pat or len(pat.stops) < 2:
                continue

            current_trip: Optional[PatternTripTimes] = None
            board_stop: Optional[str] = None
            board_time: Optional[int] = None
            board_idx: Optional[int] = None

            # Scan along pattern stops (RAPTOR-style)
            for i, sid in enumerate(pat.stops):
                arr_at_stop = earliest.get(sid, INF)

                # Can we (re)board here?
                if arr_at_stop < INF:
                    res = _find_earliest_trip(pat, i, arr_at_stop)
                    if res:
                        trip, dep_time = res
                        if current_trip is None:
                            current_trip = trip
                            board_stop = sid
                            board_time = int(dep_time)
                            board_idx = i
                        else:
                            try:
                                curr_dep_i = int(current_trip.dep[i])
                            except Exception:
                                curr_dep_i = INF
                            if int(dep_time) < curr_dep_i:
                                current_trip = trip
                                board_stop = sid
                                board_time = int(dep_time)
                                board_idx = i

                # Propagate on current boarded trip to this stop
                if current_trip is not None and board_stop is not None and board_idx is not None and i > board_idx:
                    try:
                        arr_i = int(current_trip.arr[i])
                    except Exception:
                        arr_i = INF

                    if arr_i < earliest.get(sid, INF):
                        earliest[sid] = arr_i
                        prev[sid] = BackPointer(
                            mode="bus",
                            prev_stop=board_stop,
                            pattern_id=pid,
                            trip_id=current_trip.trip_id,
                            from_index=board_idx,
                            to_index=i,
                            board_time=board_time,
                            alight_time=arr_i
                        )
                        any_update.add(sid)

        # Walk transfers after this round
        if any_update:
            changed2 = relax_walk_transfers(any_update)
            any_update |= changed2

        # Early exit if end found and no further improvements likely
        if not any_update:
            break

    best_total, best_stop, best_end_walk = best_end()
    if best_total >= INF or not best_stop:
        # If time-dependent routing fails, fallback to topology-only routing.
        target_end = end_stop_id or (end_cand[0][0] if end_cand else "")
        fb = _fallback_topology_route(start_cand, target_end) if target_end else None
        if not fb:
            return False, {"error": "Không tìm thấy lộ trình phù hợp.", "suggestion": "Hãy thử điểm khác hoặc tăng bán kính đi bộ / đổi giờ xuất phát."}

        segments: List[Dict[str, Any]] = []
        total_dist = 0.0
        walk_dist = 0.0
        bus_dist = 0.0
        total_dur_s = 0.0
        BUS_SPEED_MPS = 5.0  # ~18 km/h estimation for fallback

        # Walk from start coord to first stop
        sstop = fb["start_stop_id"]
        s_obj = STOPS[sstop]
        start_coords = [(start_lat, start_lon), (s_obj.lat, s_obj.lon)]
        wr = osrm_route("foot", start_coords)
        if wr and wr.get("path"):
            mp = wr["path"]
            dist_m = float(wr.get("distance_m") or fb.get("start_walk_m") or 0.0)
            dur_s = float(wr.get("duration_s") or (dist_m / WALK_SPEED_MPS))
        else:
            mp = start_coords
            dist_m = float(fb.get("start_walk_m") or haversine_m(start_lat, start_lon, s_obj.lat, s_obj.lon))
            dur_s = float(dist_m / WALK_SPEED_MPS)
        segments.append({
            "type": "walk",
        "route_id": "walking",
        "route_name": "Đi bộ",
        "from": "Vị trí của bạn",
        "to": (s_obj.stop_name if s_obj else sstop),
        "from_stop_id": None,
        "to_stop_id": sstop,
        "stop_ids": [sstop],
        "stops_count": 0,
        "distance": dist_m,
        "path": mp,
            "from_coord": [start_lat, start_lon],
            "to_stop": {"stop_id": sstop, "stop_name": s_obj.stop_name, "lat": s_obj.lat, "lon": s_obj.lon},
            "distance_m": dist_m,
            "duration_s": dur_s,
            "map_path": mp,
        })
        total_dist += dist_m; walk_dist += dist_m; total_dur_s += dur_s

        # Convert topology segments
        for seg in fb["segments"]:
            if seg["mode"] == "bus":
                stop_ids = seg["stop_ids"]
                coords = [(STOPS[sid].lat, STOPS[sid].lon) for sid in stop_ids if sid in STOPS]
                rr = osrm_route("driving", coords)
                if rr and rr.get("path"):
                    mp = rr["path"]
                    dist_m = float(rr.get("distance_m") or 0.0)
                    dur_s = float(rr.get("duration_s") or (dist_m / BUS_SPEED_MPS))
                else:
                    mp = coords
                    dist_m = 0.0
                    for a, b in zip(coords, coords[1:]):
                        dist_m += haversine_m(a[0], a[1], b[0], b[1])
                    dur_s = float(dist_m / BUS_SPEED_MPS) if dist_m > 0 else 0.0
                rid = seg.get("route_id", "")
                rinfo = ROUTES.get(rid)
                segments.append({
                    "type": "bus",
                    "route_id": rid,
                    "route_short_name": (rinfo.short_name if rinfo else rid),
                    "route_long_name": (rinfo.long_name if rinfo else ""),
                    "route_name": (f"Tuyến {rinfo.short_name}" if (rinfo and rinfo.short_name) else (rinfo.long_name if (rinfo and rinfo.long_name) else rid)),
                    "from": (STOPS[stop_ids[0]].stop_name if stop_ids and stop_ids[0] in STOPS else stop_ids[0]),
                    "to": (STOPS[stop_ids[-1]].stop_name if stop_ids and stop_ids[-1] in STOPS else stop_ids[-1]),
                    "stops_count": max(0, len(stop_ids) - 1),
                    "distance": dist_m,
                    "path": mp,
                    "from_stop_id": stop_ids[0],
                    "to_stop_id": stop_ids[-1],
                    "stop_ids": stop_ids,
                    "board_time": None,
                    "alight_time": None,
                    "distance_m": dist_m,
                    "duration_s": dur_s,
                    "map_path": mp,
                })
                total_dist += dist_m; bus_dist += dist_m; total_dur_s += dur_s
            else:
                stop_ids = seg["stop_ids"]
                a = STOPS[stop_ids[0]]; b = STOPS[stop_ids[-1]]
                coords = [(a.lat, a.lon), (b.lat, b.lon)]
                rr = osrm_route("foot", coords)
                if rr and rr.get("path"):
                    mp = rr["path"]
                    dist_m = float(rr.get("distance_m") or 0.0)
                    dur_s = float(rr.get("duration_s") or (dist_m / WALK_SPEED_MPS))
                else:
                    mp = coords
                    dist_m = float(haversine_m(a.lat, a.lon, b.lat, b.lon))
                    dur_s = float(dist_m / WALK_SPEED_MPS)
                segments.append({
                    "type": "walk",
                    "route_id": "walk_transfer",
                    "route_name": "Đi bộ (chuyển trạm)",
                    "from": (STOPS[stop_ids[0]].stop_name if stop_ids and stop_ids[0] in STOPS else stop_ids[0]),
                    "to": (STOPS[stop_ids[-1]].stop_name if stop_ids and stop_ids[-1] in STOPS else stop_ids[-1]),
                    "stops_count": max(0, len(stop_ids) - 1),
                    "distance": dist_m,
                    "path": mp,
                    "from_stop_id": stop_ids[0],
                    "to_stop_id": stop_ids[-1],
                    "stop_ids": stop_ids,
                    "distance_m": dist_m,
                    "duration_s": dur_s,
                    "map_path": mp,
                })
                total_dist += dist_m; walk_dist += dist_m; total_dur_s += dur_s

        # Final walk from end stop to end coord
        est_arrive = (dep0 + int(total_dur_s))
        summary = {
            "depart_time": depart_time_hhmm,
            "arrive_time": fmt_hhmm(est_arrive),
            "transfers": max(0, sum(1 for s in segments if s.get("type") == "bus") - 1),
            "distance_m": total_dist,
            "walk_m": walk_dist,
            "bus_m": bus_dist,
            "fallback": True,
            "distance": total_dist,
            "walking_distance": walk_dist,
            "bus_distance": bus_dist,
            "transfers_count": max(0, sum(1 for s in segments if s.get("type") == "bus") - 1),
            "start_stop_id": sstop,
            "start_stop_name": (STOPS[sstop].stop_name if sstop in STOPS else sstop),
            "end_stop_id": target_end,
            "end_stop_name": (STOPS[target_end].stop_name if target_end in STOPS else target_end),
        }
        return True, {"status": "ok", "summary": summary, "segments": segments}


    # Reconstruct itinerary stops
    legs: List[Dict[str, Any]] = []
    # Final walk to destination point
    end_stop_obj = STOPS[best_stop]
    # Reconstruct chain of stop legs backwards
    curr = best_stop
    while True:
        bp = prev.get(curr)
        if not bp:
            break
        if bp.mode == "bus":
            board = bp.prev_stop
            if not board:
                break
            pat = PATTERNS.get(bp.pattern_id or -1)
            rid = pat.route_id if pat else ""
            rinfo = ROUTES.get(rid)
            from_idx = bp.from_index or 0
            to_idx = bp.to_index or 0
            if pat:
                stop_slice = pat.stops[from_idx:to_idx+1]
            else:
                stop_slice = [board, curr]
            # Build coords for OSRM
            coords = [(STOPS[s].lat, STOPS[s].lon) for s in stop_slice if s in STOPS]
            osrm = osrm_route("driving", coords, snap_radius_m=OSRM_SNAP_RADIUS_M)
            if osrm:
                map_path = osrm["path"]
                dist_m = osrm["distance_m"]
                dur_s = osrm["duration_s"]
            else:
                map_path = coords
                # approximate
                dist_m = 0.0
                for (la1, lo1), (la2, lo2) in zip(coords, coords[1:]):
                    dist_m += haversine_m(la1, lo1, la2, lo2)
                dur_s = dist_m / 6.0  # fallback ~21.6km/h
            legs.append({
                "type": "bus",
                "route_id": rid,
                "route_short_name": rinfo.short_name if rinfo else "",
                "route_long_name": rinfo.long_name if rinfo else "",
                "trip_id": bp.trip_id,
                "from_stop": board,
                "to_stop": curr,
                "stop_ids": stop_slice,
                "depart_time": fmt_hhmm(bp.board_time or 0),
                "arrive_time": fmt_hhmm(bp.alight_time or 0),
                "distance_m": dist_m,
                "duration_s": dur_s,
                "map_path": map_path
            })
            curr = board
            continue

        if bp.mode == "walk":
            prev_stop = bp.prev_stop
            if not prev_stop:
                break
            a = STOPS[prev_stop]
            b = STOPS[curr]
            coords = [(a.lat, a.lon), (b.lat, b.lon)]
            osrm = osrm_route("walking", coords, snap_radius_m=25)
            map_path = osrm["path"] if osrm else coords
            legs.append({
                "type": "walk",
                "from_stop": prev_stop,
                "to_stop": curr,
                "distance_m": bp.walk_m,
                "duration_s": bp.walk_s,
                "map_path": map_path
            })
            curr = prev_stop
            continue

        if bp.mode == "walk_start":
            # final: from start coord to this start stop
            sc = bp.prev_coord
            if sc:
                a = sc
                b = STOPS[curr]
                coords = [(a[0], a[1]), (b.lat, b.lon)]
                osrm = osrm_route("walking", coords, snap_radius_m=25)
                map_path = osrm["path"] if osrm else coords
                legs.append({
                    "type": "walk",
                    "from_coord": {"lat": a[0], "lon": a[1]},
                    "to_stop": curr,
                    "distance_m": bp.walk_m,
                    "duration_s": bp.walk_s,
                    "map_path": map_path
                })
            break

        break

    # Put reconstructed legs in forward order (from start to end stop)
    legs.reverse()

    # Optional final walk from end stop to end coordinate (only if meaningful)
    if float(best_end_walk) > 30.0:
        coords_end = [(end_stop_obj.lat, end_stop_obj.lon), (end_lat, end_lon)]
        osrm_end = osrm_route("walking", coords_end, snap_radius_m=25)
        map_path_end = osrm_end["path"] if osrm_end else coords_end
        legs.append({
            "type": "walk",
            "from_stop": best_stop,
            "to_coord": {"lat": end_lat, "lon": end_lon},
            "distance_m": float(best_end_walk),
            "duration_s": float(best_end_walk) / WALK_SPEED_MPS,
            "map_path": map_path_end
        })


    # Drop negligible walking legs to reduce clutter in UI
    legs = [l for l in legs if not (l.get("type") == "walk" and float(l.get("distance_m") or 0.0) < 30.0)]

    # Summaries
    total_dist = 0.0
    walk_dist = 0.0
    bus_dist = 0.0
    transfers = 0
    bus_legs = [l for l in legs if l.get("type") == "bus"]
    transfers = max(0, len(bus_legs) - 1)
    for l in legs:
        d = float(l.get("distance_m") or 0.0)
        total_dist += d
        if l.get("type") == "walk":
            walk_dist += d
        else:
            bus_dist += d

    summary = {
        # New keys
        "depart_time": depart_time_hhmm,
        "arrive_time": fmt_hhmm(best_total),
        "transfers": transfers,
        "distance_m": total_dist,
        "walk_m": walk_dist,
        "bus_m": bus_dist,
        "stops_from": curr,   # start stop reached
        "stops_to": best_stop,

        # Backward-compatible keys used by older frontends
        "distance": total_dist,
        "walking_distance": walk_dist,
        "start_stop": curr,
        "end_stop": best_stop,
    }

    # Adapt to existing UI: use "segments"
    segments: List[Dict[str, Any]] = []
    for l in legs:
        if l["type"] == "bus":
            segments.append({
                "type": "bus",
                "route_id": l["route_id"],
                "route_short_name": l.get("route_short_name",""),
                "route_long_name": l.get("route_long_name",""),
                "route_name": (f"Tuyến {l.get('route_short_name')}" if l.get('route_short_name') else (l.get('route_long_name') or l['route_id'])),
                "from": (STOPS[l["from_stop"]].stop_name if l.get("from_stop") in STOPS else l.get("from_stop")),
                "to": (STOPS[l["to_stop"]].stop_name if l.get("to_stop") in STOPS else l.get("to_stop")),
                "stops_count": max(0, len(l.get("stop_ids") or []) - 1),
                "distance": float(l.get("distance_m") or 0.0),
                "path": l.get("map_path") or [],
                "trip_id": l.get("trip_id"),
                "from_stop": l["from_stop"],
                "to_stop": l["to_stop"],
                "stop_ids": l["stop_ids"],
                "stop_names": [(STOPS[sid].stop_name if sid in STOPS else sid) for sid in (l.get("stop_ids") or [])],
                "distance_m": l["distance_m"],
                "duration_s": l["duration_s"],
                "depart_time": l.get("depart_time"),
                "arrive_time": l.get("arrive_time"),
                "map_path": l.get("map_path") or []
            })
        else:
            seg = {
                "type": "walk",
                "route_id": "walking",
                "route_name": "Đi bộ",
                "from": (STOPS[l["from_stop"]].stop_name if (l.get("from_stop") in STOPS) else ("Vị trí của bạn" if "from_coord" in l else (l.get("from_stop") or "Vị trí của bạn"))),
                "to": (STOPS[l["to_stop"]].stop_name if (l.get("to_stop") in STOPS) else (l.get("to_stop") or "Điểm đến")),
                "stops_count": max(0, len(l.get("stop_ids") or []) - 1),
                "distance": float(l.get("distance_m") or 0.0),
                "path": l.get("map_path") or [],
                "stop_names": [],
                "distance_m": l.get("distance_m", 0.0),
                "duration_s": l.get("duration_s", 0.0),
                "map_path": l.get("map_path") or []
            }
            if "from_stop" in l:
                seg["from_stop"] = l["from_stop"]
            if "to_stop" in l:
                seg["to_stop"] = l["to_stop"]
            if "from_coord" in l:
                seg["from_coord"] = l["from_coord"]
            if "to_coord" in l:
                seg["to_coord"] = l["to_coord"]
            segments.append(seg)

    return True, {"status": "ok", "summary": summary, "segments": segments}

# -------------------- Flask app --------------------
app = Flask(__name__, static_folder=".", static_url_path="")

@app.route("/")
def index():
    # Serve frontend
    return send_from_directory(".", "bus_finder_app.html")

@app.route("/<path:path>")
def static_proxy(path):
    # Do not let this handler swallow /api/* (POST would become 405)
    if path.startswith("api/"):
        return jsonify({"status": "error", "message": "API endpoint not found"}), 404
    return send_from_directory(".", path)

@app.route("/api/ping")
def ping():
    if GTFS_OK:
        return jsonify({
            "status": "ok",
            "message": "Backend hoạt động",
            "gtfs_path": GTFS_PATH_USED,
            "stats": {
                "stops": len(STOPS),
                "routes": len(ROUTES),
                "patterns": len(PATTERNS),
                "walk_radius_m": WALK_RADIUS_M,
                "transfer_radius_m": TRANSFER_RADIUS_M,
            }
        })
    return jsonify({"status": "error", "message": GTFS_MESSAGE, "gtfs_path": GTFS_PATH_USED}), 503

@app.route("/api/stops-all")
def stops_all():
    if not GTFS_OK:
        return jsonify({"status":"error","message":GTFS_MESSAGE}), 503
    # return lightweight stops
    out = [{"stop_id": s.stop_id, "stop_name": s.stop_name, "lat": s.lat, "lon": s.lon} for s in STOPS.values()]
    return jsonify({"status":"ok","stops": out})

@app.route("/api/search-stops", methods=["POST"])
def search_stops():
    if not GTFS_OK:
        return jsonify({"status":"error","message":GTFS_MESSAGE}), 503
    data = request.get_json(silent=True) or {}
    q = str(data.get("query","") or "").strip().lower()
    limit = int(data.get("limit") or 20)
    if not q:
        return jsonify({"status":"ok","stops":[]})
    # naive substring search; fast enough for ~7k stops
    matches = []
    for s in STOPS.values():
        if q in (s.stop_name or "").lower() or q in s.stop_id.lower():
            matches.append({"stop_id": s.stop_id, "stop_name": s.stop_name, "lat": s.lat, "lon": s.lon})
            if len(matches) >= limit:
                break
    return jsonify({"status":"ok","stops": matches})



@app.route("/api/nearest-stops", methods=["POST"])
def api_nearest_stops():
    """Return nearest stops to a given lat/lon.

    Payload: {lat, lon, limit?, radius_m?}
    Response: {status:'ok', stops:[{stop_id, stop_name, lat, lon, stop_lat, stop_lon, distance_m}]}
    """
    if not GTFS_OK:
        return jsonify({"status": "error", "message": GTFS_MESSAGE}), 503

    data = request.get_json(silent=True) or {}

    def _f(v):
        try:
            return float(v)
        except Exception:
            return None

    lat = _f(data.get("lat"))
    lon = _f(data.get("lon"))
    if lon is None:
        lon = _f(data.get("lng"))
    if lat is None or lon is None:
        return jsonify({"status": "error", "message": "Thiếu lat/lon"}), 400

    limit = int(data.get("limit") or 10)
    limit = max(1, min(50, limit))

    radius_m = int(data.get("radius_m") or data.get("radius") or 1500)
    radius_m = max(100, min(20000, radius_m))

    cand = GRID.query_radius(lat, lon, radius_m, STOP_LL, limit=limit)
    if not cand:
        cand = GRID.query_radius(lat, lon, min(20000, radius_m * 2), STOP_LL, limit=limit)

    out = []
    for sid, d in cand:
        s = STOPS.get(sid)
        if not s:
            continue
        out.append({
            "stop_id": s.stop_id,
            "stop_name": s.stop_name,
            # Provide both naming styles for frontend compatibility
            "lat": float(s.lat),
            "lon": float(s.lon),
            "stop_lat": float(s.lat),
            "stop_lon": float(s.lon),
            "distance_m": float(d),
        })

    return jsonify({"status": "ok", "count": len(out), "stops": out})


@app.route("/api/find-route", methods=["POST"])
def api_find_route():
    """Find route between start and end coordinates.

    Payload:
      { start: {lat, lon}, end: {lat, lon, stop_id?}, depart_time? }
    """
    if not GTFS_OK:
        return jsonify({"status": "error", "message": GTFS_MESSAGE}), 503

    data = request.get_json(silent=True) or {}
    start = data.get("start") or {}
    end = data.get("end") or {}

    def _f(v):
        try:
            return float(v)
        except Exception:
            return None

    s_lat = _f(start.get("lat"))
    s_lon = _f(start.get("lon"))
    if s_lon is None:
        s_lon = _f(start.get("lng"))
    e_lat = _f(end.get("lat"))
    e_lon = _f(end.get("lon"))
    if e_lon is None:
        e_lon = _f(end.get("lng"))

    if s_lat is None or s_lon is None or e_lat is None or e_lon is None:
        return jsonify({"status": "error", "message": "Thiếu tọa độ start/end (lat/lon)."}), 400

    end_stop_id = end.get("stop_id") or end.get("stopId") or end.get("stopID")
    if end_stop_id is not None:
        end_stop_id = str(end_stop_id)

    depart_time = data.get("depart_time") or data.get("time") or data.get("depart") or None
    if depart_time is not None:
        depart_time = str(depart_time)

    ok, resp = plan_route(
        start_lat=float(s_lat),
        start_lon=float(s_lon),
        end_lat=float(e_lat),
        end_lon=float(e_lon),
        end_stop_id=end_stop_id,
        depart_time_hhmm=depart_time,
        max_transfers=int(data.get("max_transfers") or MAX_TRANSFERS),
    )

    if not ok:
        # Normalize error shape for frontend
        err = resp.get("error") or resp.get("message") or "Không tìm được lộ trình."
        payload = {"status": "error", "error": err}
        for k in ("suggestion", "details"):
            if k in resp:
                payload[k] = resp[k]
        return jsonify(payload), 400

    return jsonify(resp)


@app.route("/api/stops-bbox", methods=["POST","GET","OPTIONS"])
def stops_bbox():
    """Return stops within the current map bounds.

    Frontend variants observed:
      - {south, west, north, east}
      - {minLat, minLon, maxLat, maxLon}
      - {min_lat, min_lon, max_lat, max_lon}

    We accept all to avoid 400/405 issues and keep the UI responsive.
    """
    try:
        if request.method == "OPTIONS":
            return jsonify({"ok": True})

        if not GTFS_OK:
            return jsonify({"ok": False, "error": GTFS_MESSAGE}), 503

        payload = request.args if request.method == "GET" else (request.get_json(silent=True) or {})

        def _pick(*keys):
            for k in keys:
                if k in payload and payload.get(k) is not None:
                    return payload.get(k)
            return None

        south = _pick("south", "minLat", "min_lat", "minlat")
        west  = _pick("west",  "minLon", "min_lon", "minlon")
        north = _pick("north", "maxLat", "max_lat", "maxlat")
        east  = _pick("east",  "maxLon", "max_lon", "maxlon")

        if south is None or west is None or north is None or east is None:
            return jsonify({"ok": False, "error": "bbox thiếu tham số (south/west/north/east)"}), 400

        south = float(south); west = float(west); north = float(north); east = float(east)

        import math
        if not all(map(math.isfinite, [south, west, north, east])):
            return jsonify({"ok": False, "error": "bbox không hợp lệ"}), 400

        # normalize / swap if inverted
        if south > north:
            south, north = north, south
        if west > east:
            # allow dateline wrap; handled below
            pass

        # clamp to valid ranges
        south = max(-90.0, min(90.0, south))
        north = max(-90.0, min(90.0, north))
        west  = max(-180.0, min(180.0, west))
        east  = max(-180.0, min(180.0, east))

        limit = int(_pick("limit", "max", "max_results") or 3000)
        limit = max(100, min(10000, limit))
        connected_only = str(_pick("connected_only", "connectedOnly") or "1").lower() not in ("0","false","no")

        # Use in-memory GTFS structures already loaded:
        #  - STOP_LL: stop_id -> (lat, lon)
        #  - STOPS: stop_id -> Stop(...)
        #  - CONNECTED_STOP_IDS: stops served by at least one pattern

        def in_bbox(lat, lon):
            if lat < south or lat > north:
                return False
            if west <= east:
                return west <= lon <= east
            # bbox crosses dateline
            return (lon >= west) or (lon <= east)

        out = []
        count = 0
        # Fast iterate over coordinates map
        for sid, (lat, lon) in STOP_LL.items():
            if connected_only and (sid not in CONNECTED_STOP_IDS):
                continue
            if in_bbox(lat, lon):
                s = STOPS.get(sid)
                name = (s.stop_name if s else sid)
                out.append({
                    "stop_id": sid,
                    "stop_name": name,
                    # Provide both naming styles for frontend compatibility
                    "stop_lat": float(lat),
                    "stop_lon": float(lon),
                    "lat": float(lat),
                    "lon": float(lon),
                })
                count += 1
                if count >= limit:
                    break

        return jsonify({"ok": True, "count": count, "stops": out})

    except Exception as e:
        app.logger.exception("/api/stops-bbox failed")
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Save cache on exit
    try:
        # --- Run server ---
        # iPhone/Safari chỉ cho GPS + Service Worker khi chạy trong "secure context" (HTTPS).
        # Bật HTTPS bằng cách set env:
        #   USE_HTTPS=1
        # Tuỳ chọn:
        #   HTTPS_PORT=5443
        #   SSL_CONTEXT=adhoc         (tạo cert tự ký nhanh, phù hợp dev)
        #   hoặc:
        #   SSL_CERT_FILE=path/to/cert.pem
        #   SSL_KEY_FILE=path/to/key.pem
        use_https = os.getenv("USE_HTTPS", "0").strip() == "1"
        https_port = int(os.getenv("HTTPS_PORT", "5443"))
        ssl_context_mode = os.getenv("SSL_CONTEXT", "adhoc").strip().lower()
        cert_file = os.getenv("SSL_CERT_FILE", "").strip()
        key_file = os.getenv("SSL_KEY_FILE", "").strip()

        if use_https:
            if cert_file and key_file:
                ssl_ctx = (cert_file, key_file)
            else:
                # "adhoc" tạo chứng chỉ tự ký tự động (dev only)
                ssl_ctx = ssl_context_mode if ssl_context_mode else "adhoc"

            print(f"\n[BusMaps] HTTPS enabled: https://<IP>:{https_port}/")
            print("          (iPhone GPS cần mở bằng https://)")
            app.run(host="0.0.0.0", port=https_port, debug=True, ssl_context=ssl_ctx)
        else:
            print(f"\n[BusMaps] HTTP mode: http://<IP>:{DEFAULT_PORT}/")
            print("          (iPhone GPS sẽ bị chặn trên http:// — hãy bật USE_HTTPS=1)")
            app.run(host="0.0.0.0", port=DEFAULT_PORT, debug=True)

    finally:
        _save_osrm_cache(_OSRM_CACHE)