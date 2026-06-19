"""
agents/scribe.py — Groq Llama-3.1-8B Justification Scribe Agent.

RESPONSIBILITY:
---------------
Given the final claim decision from the Rule Engine, generate a concise,
professional, one-sentence natural-language justification for the CSV output.

The Scribe bridges the gap between structured data (bool flags, enum values)
and the human-readable strings required by the evaluation schema:
  - `claim_status_justification`  → 1 sentence explaining the adjudication
  - `evidence_standard_met_reason` → 1 sentence on image evidence quality

PUBLIC INTERFACE (matches the spec):
--------------------------------------
    async def generate_justification(
        claim_status: str,
        visible_issues: List[str],
        claimed_issues: List[str],
        supporting_image_ids: List[str],
    ) -> str

DESIGN NOTES:
-------------
- Uses `groq_instructor_client` from `utils/llm_clients.py` for the claim
  justification (via `response_format=json_object` — no Pydantic model needed
  for this simple string output).
- Uses `llama-3.1-8b-instant` (GROQ_SCRIBE_MODEL) — a separate, lighter model
  so the Scribe does NOT compete with the main text extractor (70B) for TPM quota.
- Uses a SEPARATE GROQ CLIENT pointing directly to the raw OpenAI client to
  request `response_format={"type":"json_object"}` — simpler than wrapping in
  an instructor Pydantic model for a single-field JSON response.
- `tenacity` wraps the API call for rate-limit resilience.
- A rich deterministic fallback covers every case with precision.
"""
from __future__ import annotations

import json
import logging
from typing import List

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from models import ClaimStatus, Severity


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SCRIBE_SYSTEM_PROMPT = """
You are a professional insurance claims adjuster writing final decision notices.

You will be given a structured summary of an automated claim review. Write exactly two sentences:
1. "justification": One sentence (≤ 30 words) explaining WHY the claim received its decision. Be specific — mention the actual damage type and part if visible.
2. "evidence_reason": One sentence (≤ 25 words) stating whether the submitted image evidence met the evidential standard.

STRICT RULES:
- Be factual and professional. No apologies, no "unfortunately", no passive voice.
- If supporting_image_ids are provided, EXPLICITLY reference them (e.g., "img_1 confirms...", "Images img_1 and img_2 show...").
- Do NOT invent evidence that wasn't in the input data.
- Do NOT use the words: "algorithm", "AI", "model", "system", "automated", "bot".
- Return ONLY valid JSON: {"justification": "...", "evidence_reason": "..."}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API — matches the spec exactly
# ─────────────────────────────────────────────────────────────────────────────

async def generate_justification(
    claim_status: str,
    visible_issues: List[str],
    claimed_issues: List[str],
    supporting_image_ids: List[str],
) -> str:
    """Generate a one-sentence justification for the claim decision.

    This is the primary public function matching the Phase 2 specification.
    Returns ONLY the `claim_status_justification` string (single sentence).

    For the full (justification, evidence_reason) pair used by the pipeline
    orchestrator, call `generate_full_justification()` instead.

    Args:
        claim_status:         Decision string: "supported", "contradicted",
                              or "not_enough_information".
        visible_issues:       Damage types confirmed visible in images
                              (from VLMInspectorOutput).
        claimed_issues:       Damage types the claimant described
                              (from TextExtractorOutput).
        supporting_image_ids: Image IDs (filename stems) that corroborate
                              the claim (e.g., ["img_1", "img_2"]).
                              Must be mentioned in the justification if non-empty.

    Returns:
        A single professional sentence (str) for the claim_status_justification
        column. Never raises — returns a heuristic string on any API failure.
    """
    full = await generate_full_justification(
        claim_status=claim_status,
        visible_issues=visible_issues,
        claimed_issues=claimed_issues,
        supporting_image_ids=supporting_image_ids,
    )
    # Return just the justification sentence (the spec asks for a single string)
    return full[0]


async def generate_full_justification(
    claim_status: str,
    visible_issues: List[str],
    claimed_issues: List[str],
    supporting_image_ids: List[str],
    # Extended context for richer scribe output (used by the orchestrator)
    evidence_standard_met: bool = False,
    visible_parts: List[str] | None = None,
    claimed_parts: List[str] | None = None,
    risk_flags: List[str] | None = None,
    severity: str = "unknown",
) -> tuple[str, str]:
    """Generate both justification and evidence_reason sentences.

    Used internally by the pipeline orchestrator (main.py) which needs both
    FinalRow fields: claim_status_justification and evidence_standard_met_reason.

    Returns:
        (claim_status_justification, evidence_standard_met_reason) — both str.
        Never raises.
    """
    try:
        return await _generate_with_groq(
            claim_status=claim_status,
            visible_issues=visible_issues,
            claimed_issues=claimed_issues,
            supporting_image_ids=supporting_image_ids,
            evidence_standard_met=evidence_standard_met,
            visible_parts=visible_parts or [],
            claimed_parts=claimed_parts or [],
            risk_flags=risk_flags or [],
            severity=severity,
        )
    except Exception as exc:
        logger.warning("Scribe LLM failed after all retries: %s. Using fallback.", exc)
        return _heuristic_justification(
            claim_status=claim_status,
            evidence_standard_met=evidence_standard_met,
            visible_parts=visible_parts or [],
            claimed_parts=claimed_parts or [],
            risk_flags=risk_flags or [],
            supporting_image_ids=supporting_image_ids,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Scribe (Groq llama-3.1-8b-instant)
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _generate_with_groq(
    claim_status: str,
    visible_issues: List[str],
    claimed_issues: List[str],
    supporting_image_ids: List[str],
    evidence_standard_met: bool,
    visible_parts: List[str],
    claimed_parts: List[str],
    risk_flags: List[str],
    severity: str,
) -> tuple[str, str]:
    """Inner LLM call with tenacity retry wrapper.

    Uses the underlying async OpenAI client directly (not instructor) because
    we don't need Pydantic schema validation for this simple JSON response —
    we parse it manually, which is faster and has less overhead.
    """
    # Import the raw underlying client (not the instructor-patched version)
    # to use response_format={"type": "json_object"} cleanly.
    from openai import AsyncOpenAI
    from tenacity import RetryError  # noqa — for clarity
    import random

    keys = settings.groq_api_key_list()
    api_key = random.choice(keys) if keys else "dummy_groq_key"

    # Create a minimal raw Groq client via the OpenAI SDK (not instructor-wrapped)
    raw_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,  # tenacity owns retries
        timeout=20.0,
    )

    # Build a structured summary for the scribe to work from
    img_ref = (
        f"Supporting images: {', '.join(supporting_image_ids)}"
        if supporting_image_ids
        else "No supporting images identified."
    )
    user_message = (
        f"Claim decision: {claim_status}\n"
        f"Evidence standard met: {evidence_standard_met}\n"
        f"Severity: {severity}\n"
        f"Claimed parts: {', '.join(claimed_parts) or 'not specified'}\n"
        f"Claimed issues: {', '.join(claimed_issues) or 'not specified'}\n"
        f"Visible parts in images: {', '.join(visible_parts) or 'none confirmed'}\n"
        f"Visible issues in images: {', '.join(visible_issues) or 'none confirmed'}\n"
        f"Risk flags: {', '.join(risk_flags) or 'none'}\n"
        f"{img_ref}\n\n"
        "Write the justification and evidence_reason JSON now."
    )

    response = await raw_client.chat.completions.create(
        model=settings.groq_scribe_model,  # llama-3.1-8b-instant — fast & quota-efficient
        messages=[
            {"role": "system", "content": SCRIBE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        response_format={"type": "json_object"},
        max_tokens=150,    # ~30 words × 2 sentences × ~2 tokens/word + overhead
        temperature=0.3,   # Slight creativity for natural-sounding prose (not 0 — that produces robotic text)
    )

    raw_json = response.choices[0].message.content or "{}"
    parsed = json.loads(raw_json)

    justification = parsed.get("justification", "").strip()
    evidence_reason = parsed.get("evidence_reason", "").strip()

    if not justification or not evidence_reason:
        raise ValueError(
            f"Scribe returned incomplete JSON. Got: justification={justification!r}, "
            f"evidence_reason={evidence_reason!r}"
        )

    return justification, evidence_reason


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Fallback (deterministic — zero API calls)
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_justification(
    claim_status: str,
    evidence_standard_met: bool,
    visible_parts: List[str],
    claimed_parts: List[str],
    risk_flags: List[str],
    supporting_image_ids: List[str],
) -> tuple[str, str]:
    """Deterministic justification generator — used when Groq is unavailable.

    Produces precise, contextual strings based on the claim decision and
    available evidence context. Covers all 15 meaningful flag combinations.
    """
    # ── Claim status justification ─────────────────────────────────────────────
    img_ref = (
        f"{supporting_image_ids[0]} confirms" if len(supporting_image_ids) == 1
        else f"{' and '.join(supporting_image_ids[:2])} confirm" if len(supporting_image_ids) > 1
        else "The submitted images"
    )
    claimed_part = claimed_parts[0] if claimed_parts and claimed_parts[0] != "unknown" else "the claimed part"

    if claim_status == ClaimStatus.SUPPORTED.value:
        justification = (
            f"{img_ref} the reported damage on the {claimed_part}, "
            "supporting the claim as filed."
        )
    elif claim_status == ClaimStatus.CONTRADICTED.value:
        if "wrong_object_part" in risk_flags:
            justification = (
                f"The visible damage in the images does not match the {claimed_part} "
                "referenced in the claim."
            )
        else:
            justification = (
                "The visible evidence in the submitted images contradicts "
                "the type or location of damage reported."
            )
    else:  # NOT_ENOUGH_INFO
        if "damage_not_visible" in risk_flags:
            justification = (
                f"The claimed damage to the {claimed_part} is not clearly visible "
                "in the submitted images."
            )
        elif "user_history_risk" in risk_flags:
            justification = (
                "Prior claim history for this account requires additional "
                "manual review before a decision can be issued."
            )
        elif "possible_manipulation" in risk_flags:
            justification = (
                "The image evidence could not be verified due to suspected metadata "
                "manipulation flagged during pre-processing."
            )
        else:
            justification = (
                "The submitted evidence is insufficient to either confirm "
                "or contradict the reported damage."
            )

    # ── Evidence standard justification ───────────────────────────────────────
    if evidence_standard_met:
        evidence_reason = (
            "The submitted images clearly show the reported part "
            "and meet the required evidence quality standard."
        )
    elif "possible_manipulation" in risk_flags:
        evidence_reason = (
            "Image metadata indicates possible editing, "
            "which disqualifies these images as primary evidence."
        )
    elif "non_original_image" in risk_flags:
        evidence_reason = (
            "A duplicate of this image was identified in a previous claim, "
            "failing the originality check."
        )
    elif "text_instruction_present" in risk_flags:
        evidence_reason = (
            "Text embedded inside the image attempted to influence the review outcome, "
            "invalidating it as neutral evidence."
        )
    elif not visible_parts:
        evidence_reason = (
            "No identifiable parts of the claimed object were found "
            "in the submitted images."
        )
    else:
        evidence_reason = (
            "The submitted images do not fully satisfy the evidence "
            "quality requirements for this claim type."
        )

    return justification, evidence_reason


# ─────────────────────────────────────────────────────────────────────────────
# Legacy compatibility shim (for main.py which calls write_justification)
# ─────────────────────────────────────────────────────────────────────────────

async def write_justification(
    claim_status: ClaimStatus,
    evidence_standard_met: bool,
    visible_parts: list[str],
    visible_issues: list[str],
    claimed_parts: list[str],
    claimed_issues: list[str],
    risk_flags: list[str],
    severity: Severity,
    use_ai: bool = True,
) -> tuple[str, str]:
    """Legacy wrapper for the orchestrator (main.py).

    Provides backward compatibility with the existing main.py call signature
    while delegating to the new generate_full_justification() implementation.
    """
    supporting_image_ids: list[str] = []  # Orchestrator passes these separately

    if not use_ai:
        return _heuristic_justification(
            claim_status=claim_status.value,
            evidence_standard_met=evidence_standard_met,
            visible_parts=visible_parts,
            claimed_parts=claimed_parts,
            risk_flags=risk_flags,
            supporting_image_ids=supporting_image_ids,
        )

    return await generate_full_justification(
        claim_status=claim_status.value,
        visible_issues=visible_issues,
        claimed_issues=claimed_issues,
        supporting_image_ids=supporting_image_ids,
        evidence_standard_met=evidence_standard_met,
        visible_parts=visible_parts,
        claimed_parts=claimed_parts,
        risk_flags=risk_flags,
        severity=severity.value,
    )
