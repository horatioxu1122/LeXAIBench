from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def progress_enabled() -> bool:
    value = os.getenv("LEGALBENCH_PROGRESS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def emit_progress(event: str, **fields: Any) -> None:
    if not progress_enabled():
        return
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    line = json.dumps(payload, sort_keys=True, default=str)
    print(line, file=sys.stderr, flush=True)

    log_path = os.getenv("LEGALBENCH_PROGRESS_LOG", "").strip()
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
