from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


@dataclass
class AdapterResult:
    participants: pd.DataFrame
    daily_observations: pd.DataFrame
    hormone_measurements: pd.DataFrame
    events: pd.DataFrame
    reference_intervals: pd.DataFrame
    cycles: pd.DataFrame
    data_quality: pd.DataFrame
    provenance: list[dict[str, str]] = field(default_factory=list)


class BaseAdapter(ABC):
    source_name: str
    access_classification = "restricted_health_data"
    shareable = False
    source_license = "not_recorded"

    @abstractmethod
    def inspect(self, source_dir: Path) -> dict[str, object]:
        """Inspect actual files and columns without inventing mappings."""

    @abstractmethod
    def convert(self, source_dir: Path) -> AdapterResult:
        """Convert inspected source data into source-preserving common tables."""

    @staticmethod
    def require_files(source_dir: Path) -> list[Path]:
        files = sorted(path for path in source_dir.rglob("*") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No source files found in {source_dir}")
        return files

    @classmethod
    def write(cls, result: AdapterResult, output_dir: Path, file_format: str = "parquet") -> dict[str, object]:
        if file_format not in {"parquet", "csv"}:
            raise ValueError("file_format must be 'parquet' or 'csv'")
        output_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "participants": result.participants,
            "daily_observations": result.daily_observations,
            "hormone_measurements": result.hormone_measurements,
            "events": result.events,
            "reference_intervals": result.reference_intervals,
            "cycles": result.cycles,
            "data_quality": result.data_quality,
        }
        suffix = ".parquet" if file_format == "parquet" else ".csv"
        targets = {name: output_dir / f"{name}{suffix}" for name in tables}
        targets["manifest"] = output_dir / "manifest.json"
        existing = [path for path in targets.values() if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing curated outputs: {names}")

        output_files: dict[str, dict[str, object]] = {}
        for name, table in tables.items():
            path = targets[name]
            if file_format == "parquet":
                try:
                    table.to_parquet(path, index=False)
                except ImportError as exc:
                    raise RuntimeError("Writing Parquet requires pyarrow; install the project data extra") from exc
            else:
                table.to_csv(path, index=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            output_files[name] = {"path": path.name, "rows": len(table), "sha256": digest}

        manifest: dict[str, object] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_name": cls.source_name,
            "access_classification": cls.access_classification,
            "shareable": cls.shareable,
            "source_license": cls.source_license,
            "file_format": file_format,
            "tables": output_files,
            "input_provenance": result.provenance,
        }
        targets["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest
