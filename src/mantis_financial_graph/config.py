from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("project", {}).get("seed", 42))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("project", {}).get("output_dir", "outputs"))


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PipelineConfig(raw=raw)
