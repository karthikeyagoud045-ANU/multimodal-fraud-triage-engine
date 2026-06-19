"""
main.py — Async Multimodal Claims Pipeline Orchestrator.

ARCHITECTURE — "Cascade" Pattern (God-Tier Stack):
====================================================
┌─────────────────────────────────────────────────────────────────────────┐
│  Per-Claim Pipeline (runs concurrently — asyncio.Semaphore(3))          │
│                                                                          │
│  Step 1 │ Image Processor    │ Resize → Rotate → phash → JPEG base64    │
│  Step 2 │ Security Pre-Flight│ EXIF forensics + OCR injection + cache   │
│  Step 3 │ Text Extractor     │ Groq Llama-3.3-70B (instructor + tenacity│
│  Step 4 │ VLM Inspector      │ Nvidia NIM Llama-3.2-90B Vision          │
│  Step 5 │ Risk Assessor      │ Python — user history + injection flags  │
│  Step 6 │ Rule Engine        │ Python — deterministic adjudication      │
│  Step 7 │ Justification Scribe│ Groq Llama-3.1-8B (fast + quota-safe)  │
│  Step 8 │ FinalRow Validator │ Pydantic — guaranteed schema compliance  │
└─────────────────────────────────────────────────────────────────────────┘

RESILIENCE:
-----------
Every step that can fail is wrapped in its own try/except. If ANY step
after Step 1 fails (including complete API outages), the claim is filled
with a safe "not_enough_information" verdict and continues. The pipeline
NEVER crashes due to a single claim failure.

RATE LIMITS:
-----------
asyncio.Semaphore(3): max 3 claims in-flight simultaneously.
Groq free tier:    30 RPM / 12,000 TPM (70B model)
Nvidia free tier:  40 RPM (vision model)
Tenacity backoff:  2s → 4s → 8s → 10s (5 attempts max per agent call)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from agents.scribe import write_justification
from agents.text_extractor import extract_claim_intent
from agents.vlm_inspector import inspect_images, fallback_inspect_images
from config import settings
from logic.risk_assessor import assess_risk_from_context
from logic.rule_engine import synthesize_final_row
from models import ClaimContext, ClaimStatus, FinalRow, Severity
from utils.data_loader import load_contexts, split_image_paths, write_output
from utils.image_processor import process_images
import utils.telemetry as telemetry


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# Path Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Try the standardised hackathon layout first; fall back to sibling `dataset/`
_DATASET_CANDIDATES = [
    PROJECT_ROOT.parent / "claims",
    PROJECT_ROOT.parent / "dataset",
    PROJECT_ROOT / "dataset",
]
DATA_ROOT = next((p for p in _DATASET_CANDIDATES if p.exists()), _DATASET_CANDIDATES[0])
CODE_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Fallback row builder — used when a claim hard-fails
# ─────────────────────────────────────────────────────────────────────────────

def _build_error_row(context: ClaimContext, error_message: str) -> Dict:
    """Build a safe fallback CSV row when a claim fails catastrophically.

    Fills every field with sensible defaults:
      - claim_status = not_enough_information
      - risk_flags   = manual_review_required
      - All other fields = their "unknown" defaults

    Args:
        context:       The ClaimContext for the failing claim.
        error_message: Short description of the failure cause.

    Returns:
        Dict with all 14 FinalRow-compatible keys populated.
    """
    return FinalRow(
        user_id=context.user_id,
        image_paths=context.image_paths,
        user_claim=context.user_claim,
        claim_object=context.claim_object,
        evidence_standard_met=False,
        evidence_standard_met_reason="Processing error — this claim requires manual review.",
        risk_flags="manual_review_required",
        issue_type="unknown",
        object_part="unknown",
        claim_status=ClaimStatus.NOT_ENOUGH_INFO,
        claim_status_justification=f"Automated processing failed: {error_message[:120]}",
        supporting_image_ids="",
        valid_image=False,
        severity=Severity.UNKNOWN,
    ).model_dump(mode="json")


# ─────────────────────────────────────────────────────────────────────────────
# Single-Claim Pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def process_claim(
    context: ClaimContext,
    dataset_root: Path,
    semaphore: asyncio.Semaphore,
    use_ai: bool,
    use_security: bool,
) -> Dict:
    """Run the full 8-step pipeline for a single claim.

    This function is designed to NEVER raise an exception. Any step that
    fails is caught, logged, and a safe fallback row is returned. This
    guarantees that asyncio.gather() always receives a result (not an
    exception) for every claim, and the output CSV always has the same
    number of rows as the input CSV.

    Args:
        context:      Loaded and validated ClaimContext for this row.
        dataset_root: Root directory for resolving relative image paths.
        semaphore:    Shared asyncio.Semaphore limiting concurrent API calls.
        use_ai:       If False, bypass all LLM/VLM calls (heuristic mode).
        use_security: If False, skip EXIF/OCR/cache pre-flight.

    Returns:
        Dict with all 14 FinalRow keys — guaranteed to be returned.
    """
    with telemetry.ClaimTimer(
        user_id=context.user_id,
        claim_object=context.claim_object.value,
        groq_model=f"{settings.groq_model} / scribe:{settings.groq_scribe_model}",
        nvidia_model=settings.nvidia_model,
    ) as timer:
        # The semaphore is acquired INSIDE the ClaimTimer so telemetry captures
        # the total wall time including any time spent waiting for the semaphore.
        async with semaphore:
            try:
                return await _run_claim_steps(
                    context, dataset_root, use_ai, use_security, timer
                )
            except Exception as exc:
                # ── Catastrophic per-claim failure handler ────────────────────
                # Log the full traceback for debugging but don't re-raise.
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "CLAIM FAILED | user_id=%s | %s\n%s",
                    context.user_id,
                    error_msg,
                    traceback.format_exc(),
                )
                timer.set_status("error")
                timer.set_error(error_msg)
                return _build_error_row(context, error_msg)


async def _run_claim_steps(
    context: ClaimContext,
    dataset_root: Path,
    use_ai: bool,
    use_security: bool,
    timer: telemetry.ClaimTimer,
) -> Dict:
    """Inner function — the actual 8-step pipeline. May raise; caller handles it."""

    # ── Step 1: Image Processing ──────────────────────────────────────────────
    # Resize to 1024px, correct EXIF rotation, compute phash, JPEG base64-encode.
    image_paths = split_image_paths(context.image_paths)
    processed_images = process_images(image_paths, dataset_root)
    logger.debug(
        "user=%s | images_processed=%d",
        context.user_id, len(processed_images)
    )

    # ── Step 2: Security Pre-Flight ───────────────────────────────────────────
    security_flags: List[str] = []
    skip_vlm = False

    if use_security:
        security_flags, skip_vlm = await asyncio.get_event_loop().run_in_executor(
            None,  # Use default ThreadPoolExecutor (OCR is CPU-bound)
            _run_security_preflight,
            processed_images,
            context.user_id,
        )
        timer.set_security_flags(security_flags)
        timer.set_skipped_vlm(skip_vlm)

        if security_flags:
            logger.info(
                "user=%s | security_flags=%s | skip_vlm=%s",
                context.user_id, security_flags, skip_vlm
            )

    # ── Step 3: Text Intent Extraction (Groq Llama-3.3-70B) ───────────────────
    t3 = time.perf_counter()
    if use_ai:
        text_output = await extract_claim_intent(
            user_claim=context.user_claim,
            claim_object=context.claim_object.value,
        )
    else:
        from models import TextExtractorOutput
        text_output = TextExtractorOutput(
            claimed_parts=["unknown"],
            claimed_issues=["unknown"],
            text_injection_detected=False
        )
    timer.log_model_call(
        provider="groq",
        function_name="extract_claim_intent",
        estimated_tokens=len(context.user_claim) // 4 + 350,
        latency_ms=(time.perf_counter() - t3) * 1000,
        success=True,
    )

    # Merge OCR-detected injection into text_output flag
    if "text_instruction_present" in security_flags:
        text_output = text_output.model_copy(update={"text_injection_detected": True})

    logger.debug(
        "user=%s | claimed_parts=%s | claimed_issues=%s | injection=%s",
        context.user_id,
        text_output.claimed_parts,
        text_output.claimed_issues,
        text_output.text_injection_detected,
    )

    # ── Step 4: Visual Inspection (Nvidia NIM Llama-3.2-90B) ──────────────────
    # Convert ProcessedImage list to the List[dict] format the agent expects.
    image_data_dicts = [
        {
            "image_id":   img.image_id,
            # Split off "data:image/jpeg;base64," prefix — the VLM receives raw b64
            "base64_data": img.data_url.split(",", 1)[1] if img.data_url else None,
            "error":      img.error,
        }
        for img in processed_images
    ]

    t4 = time.perf_counter()
    if skip_vlm or not use_ai:
        vlm_output = fallback_inspect_images(processed_images)
    else:
        vlm_output = await inspect_images(
            image_data=image_data_dicts,
            claimed_parts=text_output.claimed_parts,
            claimed_issues=text_output.claimed_issues,
            claim_object=context.claim_object.value,
        )
    timer.log_model_call(
        provider="nvidia",
        function_name="inspect_images",
        estimated_tokens=450 + 1500 * max(1, len(image_data_dicts)),
        latency_ms=(time.perf_counter() - t4) * 1000,
        success=not skip_vlm,
    )

    logger.debug(
        "user=%s | overall_severity=%s | skip_vlm=%s",
        context.user_id,
        vlm_output.overall_severity.value,
        skip_vlm,
    )

    # ── Step 5: Risk Assessment ────────────────────────────────────────────────
    # Combine user history, injection detections, and security flags into
    # the canonical risk flag list.
    visual_injection = any(
        img.visual_injection_detected for img in vlm_output.images
    )
    all_risk_flags = assess_risk_from_context(
        user_history=context.user_history,
        security_flags=security_flags,
        text_injection_detected=text_output.text_injection_detected,
        visual_injection_detected=visual_injection,
    )
    if all_risk_flags:
        logger.info(
            "user=%s | risk_flags=%s",
            context.user_id, all_risk_flags
        )

    # ── Step 6: Deterministic Rule Engine ─────────────────────────────────────
    # Pure Python — zero tokens, always fast, fully auditable.
    final_row: FinalRow = synthesize_final_row(
        context=context,
        text_output=text_output,
        vlm_output=vlm_output,
        extra_risk_flags=all_risk_flags,
    )

    # ── Step 7: Justification Scribe (Groq Llama-3.1-8B) ──────────────────────
    # Uses the lightweight 8B model to preserve 70B TPM quota.
    visible_parts = list({
        part
        for img in vlm_output.images
        for part in img.visible_parts
    })
    visible_issues = list({
        issue
        for img in vlm_output.images
        for issue in img.visible_issues
    })
    supporting_ids = [
        sid for sid in final_row.supporting_image_ids.split(";") if sid
    ]

    t7 = time.perf_counter()
    justification, evidence_reason = await write_justification(
        claim_status=final_row.claim_status,
        evidence_standard_met=final_row.evidence_standard_met,
        visible_parts=visible_parts,
        visible_issues=visible_issues,
        claimed_parts=text_output.claimed_parts,
        claimed_issues=text_output.claimed_issues,
        risk_flags=all_risk_flags,
        severity=final_row.severity,
        use_ai=use_ai,
    )
    timer.log_model_call(
        provider="groq",
        function_name="write_justification",
        estimated_tokens=120,
        latency_ms=(time.perf_counter() - t7) * 1000,
        success=True,
    )

    # Patch the rule-engine's template strings with Scribe's polished output
    final_row = final_row.model_copy(
        update={
            "claim_status_justification":  justification,
            "evidence_standard_met_reason": evidence_reason,
        }
    )

    # ── Step 8: Semantic Cache Update ─────────────────────────────────────────
    # Register the final verdict in the cache for cross-claim duplicate detection.
    if use_security:
        try:
            from security.semantic_cache import get_cache
            cache = get_cache()
            for img in processed_images:
                cache.update_status(img.phash, final_row.claim_status.value)
        except Exception:
            pass  # Cache failures must never affect the output row

    timer.set_status(final_row.claim_status.value)

    logger.info(
        "✓ user=%s | status=%-24s | severity=%s | risk=%s",
        context.user_id,
        final_row.claim_status.value,
        final_row.severity.value,
        final_row.risk_flags,
    )
    return final_row.model_dump(mode="json")


# ─────────────────────────────────────────────────────────────────────────────
# Security Pre-Flight (runs in ThreadPoolExecutor — CPU-bound)
# ─────────────────────────────────────────────────────────────────────────────

def _run_security_preflight(processed_images: list, user_id: str):
    """Synchronous security checks, designed for ThreadPoolExecutor.

    Runs EXIF forensics, semantic cache duplicate check, and OCR injection scan
    for all images in a single claim.

    Returns:
        (security_flags: List[str], skip_vlm: bool)
    """
    try:
        from security.exif_forensics import analyse_exif_from_path
        from security.ocr_sanitizer import scan_image_for_injections_from_path
        from security.semantic_cache import get_cache

        cache = get_cache()
        flags: List[str] = []
        skip_vlm = False

        for img in processed_images:
            # EXIF: check for manipulation metadata
            if img.resolved_path:
                exif_flags = analyse_exif_from_path(str(img.resolved_path))
                flags.extend(exif_flags)

            # Semantic cache: detect duplicate images across claims
            if img.phash:
                dup_flags = cache.check(img.phash, img.original_path)
                flags.extend(dup_flags)
                if "non_original_image" in dup_flags:
                    skip_vlm = True

            # OCR: detect injected text in image (skip if we're already skipping VLM)
            if img.resolved_path and not skip_vlm:
                ocr_flags = scan_image_for_injections_from_path(str(img.resolved_path))
                flags.extend(ocr_flags)

            # Register image in cache for future duplicate checks
            if img.phash:
                cache.register(img.phash, user_id)

        # Deduplicate while preserving insertion order
        seen: Dict[str, None] = {}
        deduped = [seen.setdefault(f, f) for f in flags if f not in seen]
        return deduped, skip_vlm

    except Exception as exc:
        logger.warning("Security pre-flight failed: %s — continuing without flags", exc)
        return [], False


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(args: argparse.Namespace) -> None:
    """Load data, run all claims concurrently, write output CSV and metrics.

    Orchestrates the full pipeline:
      1. Load and validate .env API keys via config.py (Settings).
      2. Load all ClaimContexts from the input CSV.
      3. Schedule all claims as concurrent asyncio tasks (rate-limited by semaphore).
      4. Write output.csv with EXACT 14-column order required by the hackathon.
      5. Write output.metrics.json with operational statistics.
    """
    load_dotenv(args.env_file)

    claims_csv  = args.claims_csv.expanduser().resolve()
    output_csv  = args.output_csv.expanduser().resolve()
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root
        else claims_csv.parent
    )

    # Configure telemetry — clears previous run data, creates output dir
    telemetry.configure(output_csv.parent)

    # Load all rows from the input CSV, merging user_history and evidence_requirements
    contexts = load_contexts(
        claims_csv=claims_csv,
        user_history_csv=args.user_history_csv,
        evidence_requirements_csv=args.evidence_requirements_csv,
    )

    use_ai       = not args.no_ai
    use_security = not args.no_security
    semaphore    = asyncio.Semaphore(args.concurrency)

    print(
        f"\n▶  Multi-Modal Fraud Triage Engine\n"
        f"   Claims:      {len(contexts)} rows  ({claims_csv.name})\n"
        f"   Concurrency: {args.concurrency} concurrent claims\n"
        f"   AI agents:   {'✅ Groq + Nvidia NIM' if use_ai else '⚠️  OFF (heuristic mode)'}\n"
        f"   Security:    {'✅ EXIF + OCR + Cache' if use_security else '⚠️  OFF'}\n"
        f"   Output:      {output_csv}\n"
    )

    start = time.perf_counter()

    # asyncio.gather() starts ALL claim coroutines simultaneously.
    # The semaphore inside process_claim() throttles actual API calls to 3 at a time.
    # return_exceptions=True ensures gather() never raises — all failures are
    # caught inside process_claim() and returned as error rows instead.
    rows = await asyncio.gather(
        *[
            process_claim(
                context=ctx,
                dataset_root=dataset_root,
                semaphore=semaphore,
                use_ai=use_ai,
                use_security=use_security,
            )
            for ctx in contexts
        ],
        return_exceptions=False,  # process_claim() itself handles all exceptions
    )

    elapsed = time.perf_counter() - start

    # Write output.csv with the exact 14-column order
    write_output(rows, output_csv)

    # ── Operational Metrics ────────────────────────────────────────────────────
    telem_summary = telemetry.summarise(output_csv.parent)
    metrics = {
        "run": {
            "claims_processed": len(rows),
            "elapsed_seconds":  round(elapsed, 3),
            "avg_seconds_per_claim": round(elapsed / len(rows), 3) if rows else 0,
            "concurrency":      args.concurrency,
            "used_ai":          use_ai,
            "used_security":    use_security,
            "claims_csv":       str(claims_csv),
            "output_csv":       str(output_csv),
        },
        "token_estimates": _estimate_token_metrics(contexts),
        "telemetry":       telem_summary,
    }
    metrics_path = output_csv.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ── Final Summary ──────────────────────────────────────────────────────────
    status_counts: Dict[str, int] = {}
    for row in rows:
        s = row.get("claim_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n{'─'*60}")
    print(f"✅  Wrote {len(rows)} rows  →  {output_csv}")
    print(f"📊  Metrics  →  {metrics_path}")
    print(f"⏱   Total: {elapsed:.1f}s  │  Avg: {elapsed/len(rows):.2f}s/claim" if rows else "")
    print(f"📋  Verdict distribution: {status_counts}")
    if telem_summary.get("errors", {}).get("count", 0) > 0:
        print(f"⚠️   Errors: {telem_summary['errors']['count']} claims failed → filled with 'not_enough_information'")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Token Estimator (for metrics report)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_token_metrics(contexts: list) -> Dict:
    """Estimate total token usage across all claims.

    Based on empirical measurements from the Phase 1 pipeline run:
      - Text extractor (70B):  ~350 tokens/claim + transcript length
      - VLM inspector (90B):   ~450 overhead + ~1500 per image
      - Scribe (8B):           ~120 tokens/claim
    """
    text_tokens   = sum(max(1, len(c.user_claim) // 4) + 350 for c in contexts)
    vision_tokens = sum(
        450 + 1500 * max(1, len(split_image_paths(c.image_paths)))
        for c in contexts
    )
    scribe_tokens = len(contexts) * 120
    total = text_tokens + vision_tokens + scribe_tokens

    return {
        "groq_text_tokens":    text_tokens + scribe_tokens,
        "nvidia_vision_tokens": vision_tokens,
        "total_estimated":     total,
        # Both APIs are free tier — no cost for this hackathon submission
        "estimated_cost_usd":  0.00,
        "cost_note": (
            "Groq free tier: unlimited tokens (rate-limited). "
            "Nvidia NIM: 1000 API credits (free tier). "
            "Zero commercial cost for this submission."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-Modal Fraud Triage Engine — Groq (Llama-3.3-70B) + "
            "Nvidia NIM (Llama-3.2-90B Vision) + Python Rule Engine."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Test run on sample data (AI enabled):
  python3 main.py --claims-csv ../../dataset/sample_claims.csv \\
                  --dataset-root ../../dataset \\
                  --output-csv ../../dataset/sample_output.csv

  # Full submission run on claims.csv:
  python3 main.py --claims-csv ../../dataset/claims.csv \\
                  --dataset-root ../../dataset \\
                  --output-csv ../../dataset/output.csv \\
                  --user-history-csv ../../dataset/user_history.csv \\
                  --evidence-requirements-csv ../../dataset/evidence_requirements.csv

  # Heuristic dry-run (no API keys needed):
  python3 main.py --no-ai --no-security \\
                  --claims-csv ../../dataset/sample_claims.csv \\
                  --output-csv ../../dataset/sample_output_dryrun.csv
        """,
    )

    parser.add_argument(
        "--claims-csv",
        type=Path,
        default=DATA_ROOT / "claims.csv",
        help="Input claims CSV (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_ROOT / "output.csv",
        help="Output CSV path (default: dataset/output.csv)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root directory for resolving relative image paths. "
            "Defaults to the directory containing the claims CSV."
        ),
    )
    parser.add_argument(
        "--user-history-csv",
        type=Path,
        default=None,
        help="user_history.csv path. If omitted, history-based risk flags are disabled.",
    )
    parser.add_argument(
        "--evidence-requirements-csv",
        type=Path,
        default=None,
        help="evidence_requirements.csv path. Optional — used by Rule Engine.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=CODE_ROOT / ".env",
        help="Path to .env file containing API keys (default: code/.env)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help=(
            "Max simultaneous claim-processing coroutines. "
            "Keep at 3 for free-tier Groq/Nvidia. Increase to 5-10 on paid plans. "
            "(default: 3)"
        ),
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help=(
            "Use heuristic extraction only — no Groq or Nvidia API calls. "
            "Useful for local dry-runs and CI validation."
        ),
    )
    parser.add_argument(
        "--no-security",
        action="store_true",
        help=(
            "Skip local security pre-flight (EXIF forensics, OCR injection scan, "
            "semantic cache). Much faster for quick testing."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for detailed per-step output.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(run_pipeline(args))
