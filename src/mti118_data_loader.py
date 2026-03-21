from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pypower.api import case118
from pypower.idx_brch import F_BUS, RATE_A, T_BUS
from pypower.idx_bus import BUS_I
from pypower.idx_gen import GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QG, QMAX, QMIN, VG


@dataclass
class MTI118Metadata:
    buses: pd.DataFrame
    generators: pd.DataFrame
    lines: pd.DataFrame
    bus_name_to_row: dict[str, int]
    bus_name_to_number: dict[str, int]


def _default_data_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _normalize_generator_token(name: str) -> str:
    match = re.match(r"([A-Za-z]+)\s*0*([0-9]+)$", name.strip())
    if not match:
        return name.strip().replace(" ", "")
    return f"{match.group(1)}{int(match.group(2))}"


def _read_series(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    value_col = "value" if "value" in df.columns else df.columns[-1]
    return df[value_col].to_numpy(dtype=float)


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _bus_token_to_number(token: str) -> int | None:
    match = re.fullmatch(r"bus0*([0-9]+)", token.strip().lower())
    if match is None:
        return None
    return int(match.group(1))


def _apply_mti_branch_limits(ppc: dict, lines: pd.DataFrame) -> None:
    """Map MTI line limits to PYPOWER branches by endpoints.

    The MTI CSV ordering is not guaranteed to match the internal PYPOWER branch
    ordering, so row-wise assignment can silently attach the wrong limits and
    make the UC model infeasible.
    """
    limits_by_edge: dict[tuple[int, int], list[float]] = {}
    for _, row in lines.iterrows():
        bus_from = _bus_token_to_number(str(row["Bus from "]))
        bus_to = _bus_token_to_number(str(row["Bus to"]))
        if bus_from is None or bus_to is None:
            continue
        limit = float(row["Max Flow (MW)"])
        if not np.isfinite(limit) or limit <= 0:
            continue
        edge = tuple(sorted((bus_from, bus_to)))
        limits_by_edge.setdefault(edge, []).append(limit)

    for branch_idx in range(ppc["branch"].shape[0]):
        bus_from = int(ppc["branch"][branch_idx, F_BUS])
        bus_to = int(ppc["branch"][branch_idx, T_BUS])
        edge = tuple(sorted((bus_from, bus_to)))
        limits = limits_by_edge.get(edge)
        if limits:
            ppc["branch"][branch_idx, RATE_A] = limits.pop(0)


def _load_fuel_prices(data_root: Path) -> dict[str, float]:
    fuels_path = data_root / "additional-files-mti-118" / "Additional Files MTI 118" / "Fuels and emission rates.csv"
    fuels = pd.read_csv(fuels_path)
    fuel_col = "Fue price ($/MMBTU)"
    return {
        str(row.iloc[0]).strip().lower(): float(row[fuel_col])
        for _, row in fuels.iterrows()
        if pd.notna(row.iloc[0]) and pd.notna(row[fuel_col])
    }


def _infer_fuel_key(generator_name: str) -> str | None:
    name = generator_name.strip()
    if name.startswith(("Solar ", "Wind ", "Hydro ", "Geo ")):
        return None
    if "Biomass" in name:
        return "biomass"
    if "Oil" in name:
        return "oil"
    if "Coal" in name:
        return "coal"
    if "NG" in name or "Natural Gas" in name:
        return "natural gas"
    return "natural gas"


def _build_case118_thermal_generators(metadata: MTI118Metadata, data_root: Path, base_mva: float) -> tuple[np.ndarray, np.ndarray, dict]:
    fuels = _load_fuel_prices(data_root)
    thermal_rows = []
    gencost_rows = []
    ramp_up_mw_per_h: list[float] = []
    ramp_down_mw_per_h: list[float] = []
    min_up_time_h: list[int] = []
    min_down_time_h: list[int] = []
    generator_names: list[str] = []

    for _, row in metadata.generators.iterrows():
        generator_name = str(row["Generator Name"]).strip()
        fuel_key = _infer_fuel_key(generator_name)
        if fuel_key is None:
            continue

        bus_name = str(row["bus of connection"]).strip()
        bus_number = metadata.bus_name_to_number.get(bus_name)
        if bus_number is None:
            continue

        pmax = _safe_float(row["Max Capacity (MW)"])
        pmin = _safe_float(row["Min Stable Level (MW)"])
        pmin = min(max(pmin, 0.0), pmax)

        startup_cost = _safe_float(row["Start Cost ($)"])
        shutdown_cost = 0.1 * startup_cost
        vom = _safe_float(row["VO&M Charge ($/MWh)"])
        fuel_price = fuels.get(fuel_key, 0.0)
        heat_rate_base = _safe_float(row["Heat Rate Base (MMBTU/hr)"])
        heat_rate_inc = _safe_float(row["Heat Rate Inc Band 1 (BTU/kWh)"])

        # Approximate a linear production cost from the first incremental heat-rate band.
        variable_fuel_cost = fuel_price * (heat_rate_inc / 1000.0)
        linear_cost = max(vom + variable_fuel_cost, 0.01)

        # Approximate no-load cost from base heat input at zero output plus VO&M at min stable level.
        no_load_cost = max(heat_rate_base * fuel_price + vom * pmin, 0.0)

        gen_row = np.zeros(21, dtype=float)
        gen_row[GEN_BUS] = bus_number
        gen_row[PG] = 0.0
        gen_row[QG] = 0.0
        gen_row[QMAX] = 0.0
        gen_row[QMIN] = 0.0
        gen_row[VG] = 1.0
        gen_row[MBASE] = base_mva
        gen_row[GEN_STATUS] = 1.0
        gen_row[PMAX] = pmax
        gen_row[PMIN] = pmin
        thermal_rows.append(gen_row)

        ramp_up_mw_per_h.append(max(_safe_float(row["Max Ramp Up (MW/min)"]) * 60.0, 0.0))
        ramp_down_mw_per_h.append(max(_safe_float(row["Max Ramp Down (MW/min)"]) * 60.0, 0.0))
        min_up_time_h.append(max(int(round(_safe_float(row["Min Up Time (h)"], default=1.0))), 1))
        min_down_time_h.append(max(int(round(_safe_float(row["Min Down Time (h)"], default=1.0))), 1))
        generator_names.append(generator_name)

        gencost_row = np.array([2.0, startup_cost, shutdown_cost, 3.0, 0.0, linear_cost, no_load_cost], dtype=float)
        gencost_rows.append(gencost_row)

    gen = np.vstack(thermal_rows) if thermal_rows else np.zeros((0, 21), dtype=float)
    gencost = np.vstack(gencost_rows) if gencost_rows else np.zeros((0, 7), dtype=float)
    meta = {
        "generator_names": generator_names,
        "ramp_up_mw_per_h": np.asarray(ramp_up_mw_per_h, dtype=float),
        "ramp_down_mw_per_h": np.asarray(ramp_down_mw_per_h, dtype=float),
        "min_up_time_h": np.asarray(min_up_time_h, dtype=int),
        "min_down_time_h": np.asarray(min_down_time_h, dtype=int),
    }
    return gen, gencost, meta


def load_mti118_metadata(data_root: Path | None = None) -> MTI118Metadata:
    data_root = Path(data_root) if data_root is not None else _default_data_root()
    addl_dir = data_root / "additional-files-mti-118" / "Additional Files MTI 118"

    buses = pd.read_csv(addl_dir / "Buses.csv")
    generators = pd.read_csv(addl_dir / "Generators.csv")
    lines = pd.read_csv(addl_dir / "Lines.csv")

    ppc = case118()
    bus_numbers = ppc["bus"][:, BUS_I].astype(int)
    bus_name_to_row = {f"bus{bus_num:03d}": idx for idx, bus_num in enumerate(bus_numbers)}
    bus_name_to_number = {f"bus{bus_num:03d}": int(bus_num) for bus_num in bus_numbers}

    return MTI118Metadata(
        buses=buses,
        generators=generators,
        lines=lines,
        bus_name_to_row=bus_name_to_row,
        bus_name_to_number=bus_name_to_number,
    )


def load_case118_ppc_with_mti_limits(data_root: Path | None = None) -> dict:
    data_root = Path(data_root) if data_root is not None else _default_data_root()
    metadata = load_mti118_metadata(data_root)
    ppc = case118()
    _apply_mti_branch_limits(ppc, metadata.lines)
    gen, gencost, uc_meta = _build_case118_thermal_generators(metadata, data_root, float(ppc["baseMVA"]))
    ppc["gen"] = gen
    ppc["gencost"] = gencost
    ppc["uc_ramp_up_mw_per_h"] = uc_meta["ramp_up_mw_per_h"]
    ppc["uc_ramp_down_mw_per_h"] = uc_meta["ramp_down_mw_per_h"]
    ppc["uc_min_up_time_h"] = uc_meta["min_up_time_h"]
    ppc["uc_min_down_time_h"] = uc_meta["min_down_time_h"]
    ppc["uc_generator_names"] = uc_meta["generator_names"]

    return ppc


def _build_load_matrix(metadata: MTI118Metadata, data_root: Path, market: str) -> np.ndarray:
    load_dir = data_root / "input-files" / "Input files" / market / "Load"
    region_series = {
        "R1": _read_series(load_dir / "LoadR1DA.csv"),
        "R2": _read_series(load_dir / "LoadR2DA.csv"),
        "R3": _read_series(load_dir / "LoadR3DA.csv"),
    }

    horizon = len(next(iter(region_series.values())))
    nb = len(metadata.bus_name_to_row)
    load_matrix = np.zeros((nb, horizon), dtype=float)

    for _, row in metadata.buses.iterrows():
        bus_name = str(row["Bus Name"]).strip()
        region = str(row["Region"]).strip()
        factor = float(row["Load Participation Factor"])
        bus_idx = metadata.bus_name_to_row.get(bus_name)
        if bus_idx is None or region not in region_series:
            continue
        load_matrix[bus_idx, :] = region_series[region] * factor

    return load_matrix


def _renewable_series_path(base_dir: Path, generator_name: str, market: str) -> Path | None:
    token = _normalize_generator_token(generator_name)
    name = generator_name.strip()

    if name.startswith("Solar "):
        return base_dir / market / "Solar" / f"{token}DA.csv"
    if name.startswith("Wind "):
        return base_dir / market / "Wind" / f"{token}DA.csv"
    if name.startswith("Hydro "):
        return base_dir / "Hydro" / f"{name}.csv"
    return None


def _is_renewable_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in ("Solar ", "Wind ", "Hydro ", "Geo "))


def _build_renewable_matrix(
    metadata: MTI118Metadata,
    data_root: Path,
    market: str,
    horizon: int,
    missing_policy: str = "drop",
) -> tuple[np.ndarray, dict]:
    input_dir = data_root / "input-files" / "Input files"
    nb = len(metadata.bus_name_to_row)
    renewable_matrix = np.zeros((nb, horizon), dtype=float)
    used_generators: list[str] = []
    skipped_generators: list[str] = []

    if missing_policy != "drop":
        raise ValueError(f"Unsupported missing_policy={missing_policy!r}; only 'drop' is currently supported")

    for _, row in metadata.generators.iterrows():
        generator_name = str(row["Generator Name"]).strip()
        if not _is_renewable_name(generator_name):
            continue

        bus_name = str(row["bus of connection"]).strip()
        bus_idx = metadata.bus_name_to_row.get(bus_name)
        if bus_idx is None:
            continue

        series_path = _renewable_series_path(input_dir, generator_name, market)

        if series_path is not None and series_path.exists():
            series = _read_series(series_path)
            if len(series) < horizon:
                raise ValueError(f"Time series {series_path} shorter than expected horizon {horizon}")
            renewable_matrix[bus_idx, :] += series[:horizon]
            used_generators.append(generator_name)
        else:
            skipped_generators.append(generator_name)

    summary = {
        "used_generators": used_generators,
        "skipped_generators": skipped_generators,
        "used_count": len(used_generators),
        "skipped_count": len(skipped_generators),
    }
    return renewable_matrix, summary


def build_case118_daily_samples(
    data_root: Path | None = None,
    market: str = "DA",
    horizon: int = 24,
    max_days: int | None = None,
    missing_policy: str = "drop",
) -> list[dict]:
    """Build day-level case118 scenarios from MTI118 data files."""
    data_root = Path(data_root) if data_root is not None else _default_data_root()
    metadata = load_mti118_metadata(data_root)

    load_matrix = _build_load_matrix(metadata, data_root, market)
    renewable_matrix, renewable_summary = _build_renewable_matrix(
        metadata,
        data_root,
        market,
        load_matrix.shape[1],
        missing_policy=missing_policy,
    )

    total_horizon = load_matrix.shape[1]
    n_days = total_horizon // horizon
    if max_days is not None:
        n_days = min(n_days, max_days)

    samples: list[dict] = []
    for day_idx in range(n_days):
        start = day_idx * horizon
        end = start + horizon
        load_day = load_matrix[:, start:end]
        renewable_day = renewable_matrix[:, start:end]
        samples.append(
            {
                "sample_id": day_idx,
                "load_data": load_day,
                "renewable_data": renewable_day,
                "pd_data": load_day,
                "renewable_summary": renewable_summary,
                "active_set": [],
            }
        )

    return samples

