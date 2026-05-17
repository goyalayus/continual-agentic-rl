"""Small JSONL logger for harness internals.

The files this writes are intentionally boring JSONL. They can be uploaded to S3
as-is later, and they are easy to inspect locally while we are still moving fast.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HarnessLogger:
    def __init__(self, log_dir: Path | str | None = None, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime(
            "run_%Y%m%d_%H%M%S_"
        ) + uuid.uuid4().hex[:8]
        base_dir = (
            Path(log_dir)
            if log_dir is not None
            else Path("benchmark_evaluation/custom_harness_runs")
        )
        self.run_dir = base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    def log(self, event_type: str, **payload: Any) -> None:
        row = {
            "run_id": self.run_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_seconds": time.monotonic(),
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return path

    def sync_to_s3(self, s3_uri: str, *, strict: bool = False) -> bool:
        """Upload this run directory with the AWS CLI if it is configured."""
        destination = s3_uri.rstrip("/") + f"/{self.run_id}/"
        self.log("s3_sync_start", destination=destination)
        try:
            result = subprocess.run(
                ["aws", "s3", "sync", str(self.run_dir), destination],
                check=False,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            self.log(
                "s3_sync_failed",
                destination=destination,
                error="AWS CLI executable not found",
            )
            if strict:
                raise RuntimeError("aws CLI is not installed or not on PATH") from exc
            return False
        self.log(
            "s3_sync_done",
            destination=destination,
            returncode=result.returncode,
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-4000:],
        )
        if result.returncode != 0:
            self.log(
                "s3_sync_failed",
                destination=destination,
                returncode=result.returncode,
                stderr=result.stderr[-4000:],
            )
            if strict:
                raise RuntimeError(
                    "aws s3 sync failed. Check AWS credentials and AWS CLI setup."
                )
            return False
        final_events_upload = subprocess.run(
            ["aws", "s3", "cp", str(self.events_path), destination + "events.jsonl"],
            check=False,
            text=True,
            capture_output=True,
        )
        if final_events_upload.returncode != 0:
            self.log(
                "s3_sync_failed",
                destination=destination,
                phase="final_events_upload",
                returncode=final_events_upload.returncode,
                stderr=final_events_upload.stderr[-4000:],
            )
            if strict:
                raise RuntimeError(
                    "aws s3 cp for final events.jsonl failed. "
                    "The run artifacts may have uploaded, but the final S3 status event is local only."
                )
            return False
        self.log(
            "s3_final_events_upload_done",
            destination=destination,
            returncode=final_events_upload.returncode,
        )
        return True
