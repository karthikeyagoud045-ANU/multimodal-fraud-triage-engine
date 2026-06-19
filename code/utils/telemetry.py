"""
utils/telemetry.py — Thread-Safe Operational Telemetry Logger.

Records per-claim and per-model-call metadata to a local JSONL file.
Designed for hackathon evaluation reports — provides latency, token usage,
and model call statistics without any external observability service.

THREAD SAFETY:
--------------
asyncio.gather() runs claims concurrently. Multiple coroutines may call
_append() nearly simultaneously. We use threading.Lock (not asyncio.Lock)
because file I/O is synchronous — a threading lock correctly serialises
concurrent file writes from different async tasks running in the same thread.

JSONL FORMAT:
-------------
Each record is a self-contained JSON object on one line. This means:
  - Partial writes (a crash mid-run) don't corrupt earlier records.
  - The file can be streamed line-by-line without loading all records.
  - Trivially loaded with: pd.read_json("telemetry.jsonl", lines=True)
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Module-Level State
# ─────────────────────────────────────────────────────────────────────────────

_LOG_PATH: Optional[Path] = None
_WRITE_LOCK = threading.Lock()  # Serialises concurrent file appends


def configure(output_dir: Path) -> None:
    """Initialise telemetry at pipeline startup.

    Must be called ONCE before any ClaimTimer or log_model_call usage.
    Creates the output directory if it doesn't exist. Clears any previous
    telemetry.jsonl from earlier runs so the report reflects only the
    current run.

    Args:
        output_dir: Directory where telemetry.jsonl will be written.
                    Typically the same directory as output.csv.
    """
    global _LOG_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = output_dir / "telemetry.jsonl"
    # Clear previous run's telemetry so we don't aggregate across runs
    if _LOG_PATH.exists():
        _LOG_PATH.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Per-Claim Timer (context manager)
# ─────────────────────────────────────────────────────────────────────────────

class ClaimTimer:
    """Context manager that times a single claim and writes a telemetry record.

    Usage:
        with ClaimTimer(user_id="user_001", ...) as timer:
            # ... process the claim ...
            timer.set_status("supported")
            timer.set_security_flags(["possible_manipulation"])
            timer.log_model_call("groq", "extract_claim_intent", tokens=350, success=True)
        # On __exit__, writes the full record to telemetry.jsonl automatically.

    The record format matches the schema in `summarise()` below.
    """

    def __init__(
        self,
        user_id: str,
        claim_object: str,
        groq_model: str,
        nvidia_model: str,
    ) -> None:
        self.user_id = user_id
        self.claim_object = claim_object
        self.groq_model = groq_model
        self.nvidia_model = nvidia_model
        # Mutable state — set via helper methods during claim processing
        self._start: float = 0.0
        self.claim_status: str = "pending"
        self.security_flags: List[str] = []
        self.skipped_vlm: bool = False
        self.model_calls: List[Dict] = []  # One entry per LLM/VLM API call
        self.error: Optional[str] = None

    # ── Context Manager Protocol ───────────────────────────────────────────────

    def __enter__(self) -> "ClaimTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)

        # Capture exception details if the claim failed
        if exc_type is not None and self.error is None:
            self.error = f"{exc_type.__name__}: {exc_val}"

        record = {
            "user_id": self.user_id,
            "claim_object": self.claim_object,
            "claim_status": self.claim_status,
            "latency_ms": elapsed_ms,
            "groq_model": self.groq_model,
            "nvidia_model": self.nvidia_model,
            "security_flags": self.security_flags,
            "skipped_vlm": self.skipped_vlm,
            "model_calls": self.model_calls,
            "error": self.error,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _append(record)
        # Return False — don't suppress exceptions; let the orchestrator handle them
        return False

    # ── Setter Methods (called during claim processing) ────────────────────────

    def set_status(self, status: str) -> None:
        """Set the final claim status (call after rule engine runs)."""
        self.claim_status = status

    def set_security_flags(self, flags: List[str]) -> None:
        """Record the security pre-flight flags for this claim."""
        self.security_flags = list(flags)

    def set_skipped_vlm(self, skipped: bool) -> None:
        """Mark whether the VLM call was skipped (e.g., duplicate image)."""
        self.skipped_vlm = skipped

    def set_error(self, error: str) -> None:
        """Record a processing error for this claim."""
        self.error = error

    def log_model_call(
        self,
        provider: str,
        function_name: str,
        estimated_tokens: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record one LLM/VLM API call for this claim.

        Args:
            provider:         "groq" or "nvidia".
            function_name:    Name of the agent function (e.g., "extract_claim_intent").
            estimated_tokens: Rough token count for cost estimation.
            latency_ms:       Wall-clock time for this specific API call.
            success:          False if the call failed (even after retries).
        """
        self.model_calls.append({
            "provider":          provider,
            "function":          function_name,
            "estimated_tokens":  estimated_tokens,
            "latency_ms":        round(latency_ms, 1),
            "success":           success,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Internal File Writer
# ─────────────────────────────────────────────────────────────────────────────

def _append(record: dict) -> None:
    """Thread-safely append one JSON record to the telemetry file.

    Uses a threading.Lock because async tasks run in a single OS thread —
    the lock prevents interleaved writes when gather() fires multiple
    coroutines that all exit their ClaimTimer context nearly simultaneously.

    Failures are silently swallowed — telemetry MUST NEVER crash the pipeline.
    """
    if _LOG_PATH is None:
        return
    try:
        with _WRITE_LOCK:
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Summariser (called at end of run to produce metrics.json)
# ─────────────────────────────────────────────────────────────────────────────

def summarise(output_dir: Path) -> dict:
    """Read telemetry.jsonl and return a summary dict for the metrics file.

    Computes:
        - Per-claim latency statistics (avg, min, max, p90)
        - Claim status distribution
        - Total model call counts and estimated token usage per provider
        - Security flag frequency table
        - Error count and list

    Returns:
        A nested dict ready to be embedded in output.metrics.json.
        Returns {} if no telemetry data exists.
    """
    log_path = output_dir / "telemetry.jsonl"
    if not log_path.exists():
        return {}

    records: List[dict] = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return {}

    # ── Latency statistics ─────────────────────────────────────────────────────
    latencies = sorted(r["latency_ms"] for r in records)
    n = len(latencies)
    p90_idx = int(n * 0.9)
    avg_latency = round(sum(latencies) / n, 1) if n else 0

    # ── Claim status distribution ──────────────────────────────────────────────
    statuses = [r.get("claim_status", "unknown") for r in records]
    status_dist = {s: statuses.count(s) for s in sorted(set(statuses))}

    # ── Model call statistics ──────────────────────────────────────────────────
    all_calls = [call for r in records for call in r.get("model_calls", [])]
    groq_calls = [c for c in all_calls if c.get("provider") == "groq"]
    nvidia_calls = [c for c in all_calls if c.get("provider") == "nvidia"]
    groq_tokens = sum(c.get("estimated_tokens", 0) for c in groq_calls)
    nvidia_tokens = sum(c.get("estimated_tokens", 0) for c in nvidia_calls)
    failed_calls = [c for c in all_calls if not c.get("success", True)]

    # ── Security flag frequency ────────────────────────────────────────────────
    all_flags = [f for r in records for f in r.get("security_flags", [])]
    flag_freq: Dict[str, int] = {}
    for flag in all_flags:
        flag_freq[flag] = flag_freq.get(flag, 0) + 1

    # ── Error summary ─────────────────────────────────────────────────────────
    errors = [r.get("error") for r in records if r.get("error")]

    return {
        "total_claims": n,
        "latency": {
            "avg_ms":  avg_latency,
            "min_ms":  latencies[0] if latencies else 0,
            "max_ms":  latencies[-1] if latencies else 0,
            "p90_ms":  latencies[p90_idx] if latencies else 0,
        },
        "claim_status_distribution": status_dist,
        "model_calls": {
            "groq_total":        len(groq_calls),
            "nvidia_total":      len(nvidia_calls),
            "failed_total":      len(failed_calls),
            "groq_tokens_est":   groq_tokens,
            "nvidia_tokens_est": nvidia_tokens,
        },
        "security": {
            "flags_raised":      len(all_flags),
            "flag_frequency":    flag_freq,
            "vlm_skipped_count": sum(1 for r in records if r.get("skipped_vlm")),
        },
        "errors": {
            "count":   len(errors),
            "samples": errors[:5],  # First 5 error messages for debugging
        },
    }
