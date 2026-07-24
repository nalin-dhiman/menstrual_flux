from __future__ import annotations

from pathlib import Path

from digital_twin.data_adapters.base import AdapterResult, BaseAdapter


class BBTAdapter(BaseAdapter):
    source_name = "basal_body_temperature"

    def inspect(self, source_dir: Path) -> dict[str, object]:
        files = self.require_files(source_dir)
        return {
            "source": self.source_name,
            "files": [str(x) for x in files],
            "required_review": ["licence", "participant_linkage", "label_definition", "treatment_exclusions", "timestamps", "measurement_protocol"],
        }

    def convert(self, source_dir: Path) -> AdapterResult:
        self.inspect(source_dir)
        raise NotImplementedError("Define a release-specific map only after licence, linkage, labels, and temperature protocol are verified")
