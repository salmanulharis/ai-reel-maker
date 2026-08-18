from __future__ import annotations

import json
import time
from pathlib import Path


def write_log(job_dir: Path, message: str, level: str = "INFO") -> dict:
    payload = {"level": level, "message": message, "ts": time.time()}
    path = Path(job_dir) / "job.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return payload
