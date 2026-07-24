from __future__ import annotations

from pathlib import Path

from digital_twin.data_adapters.base import AdapterResult, BaseAdapter


class BioCycleAdapter(BaseAdapter):
    source_name = "BioCycle"

    def inspect(self, source_dir: Path) -> dict[str, object]:
        files = self.require_files(source_dir)
        return {"source": self.source_name, "files": [str(x) for x in files], "phase_labels": "must_remain_intervals"}

    def convert(self, source_dir: Path) -> AdapterResult:
        self.inspect(source_dir)
        raise NotImplementedError("An exact release-specific map is required; clinic visits must not be fabricated as exact ovulation dates")
