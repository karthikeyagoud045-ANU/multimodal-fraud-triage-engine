"""
logic/risk_assessor.py — Standalone Risk & Fraud Signal Assessor.

This module is the ONLY place where risk scoring logic lives. It takes raw
signals from every layer of the pipeline (user history, security modules,
AI agent injection flags, image hashes) and synthesises them into a clean,
deduplicated list of risk flag strings.

WHY A SEPARATE MODULE?
-----------------------
The Rule Engine (rule_engine.py) owns claim adjudication logic.
The Risk Assessor owns fraud signal aggregation.
Keeping them separate means:
  - Risk logic can be tested independently with mock inputs.
  - New fraud signals (e.g., GPS spoofing, device fingerprinting) can be added
    without touching the Rule Engine decision tree.
  - Auditors can review fraud detection logic in isolation.

OUTPUT CONTRACT:
----------------
The returned list contains ONLY these canonical flag strings:
  user_history_risk       — Claimant has elevated rejected/manual-review history.
  manual_review_required  — Claim must be escalated to a human adjudicator.
  text_instruction_present — Adversarial override detected in text or image.
  non_original_image      — Image is a known duplicate of a previously submitted photo.
  possible_manipulation   — Image metadata or content suggests digital editing.

Returning "none" is handled by the Rule Engine — this module returns [].
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Threshold Constants
# ─────────────────────────────────────────────────────────────────────────────

# How many rejected claims triggers the user_history_risk flag.
# Set to 1 — ANY prior rejection is a signal worth flagging for review.
REJECTED_CLAIM_THRESHOLD: int = 1

# How many manual-review claims triggers the flag.
MANUAL_REVIEW_THRESHOLD: int = 1

# Standard field names for rejected/manual columns in user_history.csv.
# All normalised to lowercase for case-insensitive matching.
HISTORY_REJECTED_FIELDS: frozenset[str] = frozenset({
    "rejected_claim", "rejected_claims", "rejections",
    "claim_rejected", "claims_rejected", "num_rejected",
})
HISTORY_MANUAL_FIELDS: frozenset[str] = frozenset({
    "manual_review_claim", "manual_review_claims", "manual_reviews",
    "manual_review", "num_manual_reviews", "manual_flag",
})


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def assess_risk(
    user_history: Optional[Dict],
    current_flags: List[str],
    text_injection: bool,
    visual_injection: bool,
    image_hashes: Optional[List[str]] = None,
    known_bad_hashes: Optional[Set[str]] = None,
) -> List[str]:
    """Aggregate all fraud signals into a deduplicated risk flag list.

    This is the sole public function of this module. Call it once per claim
    after the security pre-flight and AI agents have run.

    Args:
        user_history:      Dict from user_history.csv for this user_id.
                           May be None if the file is not provided or the
                           user has no history record.
        current_flags:     Flags already raised by security modules (EXIF,
                           OCR, semantic cache). These are passed through
                           verbatim and deduplicated with newly assessed flags.
        text_injection:    True if text_extractor detected adversarial commands
                           in the transcript (TextExtractorOutput.text_injection_detected).
        visual_injection:  True if vlm_inspector detected text-based injection
                           inside any submitted image (ImageAnalysis.visual_injection_detected).
        image_hashes:      List of phash hex strings for all submitted images.
                           Used for cross-claim duplicate detection when
                           known_bad_hashes is provided.
        known_bad_hashes:  Set of phash strings from previously rejected claims.
                           When an image hash matches this set, non_original_image
                           is flagged. Typically populated by semantic_cache.py.

    Returns:
        A deduplicated list of canonical risk flag strings.
        Returns [] (empty list) when no risks are detected — the Rule Engine
        converts this to "none" in the CSV output.
    """
    flags: List[str] = list(current_flags)  # Start from upstream security flags

    # ── Signal 1: User History Risk ───────────────────────────────────────────
    history_flags = _assess_user_history(user_history)
    flags.extend(history_flags)
    if history_flags:
        logger.debug(
            "User history risk detected: %s → flags=%s",
            user_history, history_flags
        )

    # ── Signal 2: Prompt Injection (Text or Visual) ───────────────────────────
    if text_injection or visual_injection:
        if "text_instruction_present" not in flags:
            flags.append("text_instruction_present")
        logger.debug(
            "Injection detected: text=%s, visual=%s", text_injection, visual_injection
        )

    # ── Signal 3: Known Bad Image Hash (Cross-Claim Duplicate) ───────────────
    if image_hashes and known_bad_hashes:
        for phash in image_hashes:
            if phash and phash in known_bad_hashes:
                if "non_original_image" not in flags:
                    flags.append("non_original_image")
                logger.debug("Duplicate image hash detected: %s", phash)
                break  # One match is enough to flag the claim

    return _deduplicate(flags)


def assess_risk_from_context(
    user_history: Optional[Dict],
    security_flags: List[str],
    text_injection_detected: bool,
    visual_injection_detected: bool,
) -> List[str]:
    """Simplified wrapper for the main orchestrator.

    Accepts the output shapes directly from the orchestrator context
    without needing to pass image hash sets (the semantic cache handles
    that separately via its own flag injection into security_flags).

    Args:
        user_history:             Dict from ClaimContext.user_history.
        security_flags:           List of flags from EXIF/OCR/cache pre-flight.
        text_injection_detected:  From TextExtractorOutput.text_injection_detected.
        visual_injection_detected: True if any ImageAnalysis.visual_injection_detected.

    Returns:
        Deduplicated list of all risk flags for this claim.
    """
    return assess_risk(
        user_history=user_history,
        current_flags=security_flags,
        text_injection=text_injection_detected,
        visual_injection=visual_injection_detected,
        image_hashes=None,
        known_bad_hashes=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assess_user_history(history: Optional[Dict]) -> List[str]:
    """Derive risk flags from the user_history.csv row for this claimant.

    Returns:
        ["user_history_risk", "manual_review_required"] if the history
        indicates elevated risk, otherwise [].
    """
    if not history:
        return []

    rejected_count = 0
    manual_count = 0

    for raw_key, raw_value in history.items():
        # Normalise key to lowercase for case-insensitive matching
        key = str(raw_key).strip().lower()

        if key in HISTORY_REJECTED_FIELDS:
            rejected_count += _parse_count(raw_value)

        if key in HISTORY_MANUAL_FIELDS:
            manual_count += _parse_count(raw_value)

    flags: List[str] = []
    if rejected_count >= REJECTED_CLAIM_THRESHOLD:
        flags.append("user_history_risk")
        flags.append("manual_review_required")
    elif manual_count >= MANUAL_REVIEW_THRESHOLD:
        # Manual reviews without rejections are still worth flagging,
        # but less severely — only add manual_review_required.
        flags.append("manual_review_required")

    return flags


def _parse_count(value) -> int:
    """Parse a count value from a user_history field.

    Handles:
        - Integer / float strings: "2", "2.0" → 2
        - Boolean strings: "true", "yes", "y" → 1 (treated as count of 1)
        - NaN / empty / None → 0
    """
    if value is None:
        return 0
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "null", ""}:
        return 0
    if s in {"true", "yes", "y"}:
        return 1
    if s in {"false", "no", "n"}:
        return 0
    try:
        return int(float(s))
    except (ValueError, OverflowError):
        return 0


def _deduplicate(flags: List[str]) -> List[str]:
    """Remove duplicates while preserving insertion order."""
    seen: Dict[str, None] = {}
    return [seen.setdefault(f, f) for f in flags if f not in seen]
