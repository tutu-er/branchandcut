from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pypower.api import case118
from pypower.idx_brch import RATE_A
from pypower.idx_bus import BUS_I


@dataclass
class MTI118Metadata:
    buses: pd.DataFrame
    generators: pd.DataFrame
    lines: pd.DataFrame
    bus_name_to_row: dict[str, int]


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


def load_mti118_metadata(data_root: Path | None = None) -> MTI118Metadata:
    data_root = Path(data_root) if data_root is not None else _default_data_root()
    addl_dir = data_root / "additional-files-mti-118" / "Additional Files MTI 118"

    buses = pd.read_csv(addl_dir / "Buses.csv")
    generators = pd.read_csv(addl_dir / "Generators.csv")
    lines = pd.read_csv(addl_dir / "Lines.csv")

    ppc = case118()
    bus_numbers = ppc["bus"][:, BUS_I].astype(int)
    bus_name_to_row = {f"bus{bus_num:03d}": idx for idx, bus_num in enumerate(bus_numbers)}

    return MTI118Metadata(
        buses=buses,
        generators=generators,
        lines=lines,
        bus_name_to_row=bus_name_to_row,
    )


def load_case118_ppc_with_mti_limits(data_root: Path | None = None) -> dict:
    data_root = Path(data_root) if data_root is not None else _default_data_root()
    metadata = load_mti118_metadata(data_root)
    ppc = case118()

    if len(metadata.lines) == ppc["branch"].shape[0]:
        ppc["branch"][:, RATE_A] = metadata.lines["Max Flow (MW)"].to_numpy(dtype=float)

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

