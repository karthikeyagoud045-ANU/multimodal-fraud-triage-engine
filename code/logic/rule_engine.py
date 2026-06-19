from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from models import (
    ClaimContext,
    ClaimObject,
    ClaimStatus,
    FinalRow,
    ImageAnalysis,
    IssueType,
    Severity,
    TextExtractorOutput,
    VLMInspectorOutput,
)


def synthesize_claim_verdict(
    text_intent: TextExtractorOutput,
    vlm_output: VLMInspectorOutput,
    claim_object: str,
    processed_images: list,
    extra_risk_flags: Optional[List[str]] = None,
) -> Dict:
    """New Phase-3 entrypoint: synthesise AI outputs into a FinalRow-compatible dict.

    This function wraps the existing synthesize_final_row() by building a
    minimal ClaimContext from the provided inputs and delegating to the
    full implementation. Returns a dict (not FinalRow) so the orchestrator
    can validate it against FinalRow before writing.

    Args:
        text_intent:      Output from extract_claim_intent().
        vlm_output:       Output from inspect_images().
        claim_object:     Object category string: "car", "laptop", or "package".
        processed_images: List of ProcessedImage objects from image_processor.
        extra_risk_flags: Additional flags from risk_assessor / security modules.

    Returns:
        Dict matching all 14 FinalRow fields.
    """
    # Build a minimal ClaimContext for the existing synthesize_final_row
    try:
        obj_enum = ClaimObject(claim_object)
    except ValueError:
        obj_enum = ClaimObject.CAR

    # Build image_paths string from processed images
    image_paths_str = ";".join(img.original_path for img in processed_images)

    context = ClaimContext(
        user_id="__verdict_only__",
        image_paths=image_paths_str,
        user_claim="",
        claim_object=obj_enum,
    )

    final_row = synthesize_final_row(
        context=context,
        text_output=text_intent,
        vlm_output=vlm_output,
        extra_risk_flags=extra_risk_flags or [],
    )
    return final_row.model_dump(mode="json")


INVALID_IMAGE_FLAGS = {
    "cropped_or_obstructed",
    "possible_manipulation",
    "non_original_image",
    "wrong_object",
    "wrong_object_part",
}
QUALITY_BLOCKING_FLAGS = INVALID_IMAGE_FLAGS | {"damage_not_visible", "wrong_angle"}
HISTORY_REJECTED_FIELDS = {"rejected_claim", "rejected_claims", "rejections"}
HISTORY_MANUAL_FIELDS = {"manual_review_claim", "manual_review_claims", "manual_reviews"}


def synthesize_final_row(
    context: ClaimContext,
    text_output: TextExtractorOutput,
    vlm_output: VLMInspectorOutput,
    extra_risk_flags: Optional[List[str]] = None,
) -> FinalRow:
    visible_parts = _dedupe(_flatten(image.visible_parts for image in vlm_output.images))
    visible_issues = _dedupe(_flatten(image.visible_issues for image in vlm_output.images))
    claimed_parts = _clean_values(text_output.claimed_parts)
    claimed_issues = _clean_values(text_output.claimed_issues)

    risk_flags = _assemble_risk_flags(
        context=context,
        text_output=text_output,
        vlm_output=vlm_output,
        claimed_parts=claimed_parts,
        claimed_issues=claimed_issues,
        visible_parts=visible_parts,
        visible_issues=visible_issues,
        extra_risk_flags=extra_risk_flags or [],
    )
    valid_image = not any(flag in INVALID_IMAGE_FLAGS for flag in risk_flags)
    all_parts_visible = _all_claimed_visible(claimed_parts, visible_parts)
    issues_match = _issues_match(claimed_issues, visible_issues)
    any_damage_visible = _any_damage_visible(visible_issues)

    if "claim_mismatch" in risk_flags:
        claim_status = ClaimStatus.CONTRADICTED
    elif "damage_not_visible" in risk_flags or "wrong_object_part" in risk_flags:
        claim_status = ClaimStatus.NOT_ENOUGH_INFO
    elif not valid_image and not any_damage_visible:
        claim_status = ClaimStatus.NOT_ENOUGH_INFO
    elif all_parts_visible and issues_match:
        claim_status = ClaimStatus.SUPPORTED
    else:
        claim_status = ClaimStatus.NOT_ENOUGH_INFO

    evidence_standard_met = bool(
        all_parts_visible
        and valid_image
        and not any(flag in QUALITY_BLOCKING_FLAGS for flag in risk_flags)
    )

    issue_type = _join_or_unknown(visible_issues or claimed_issues)
    object_part = _join_or_unknown(visible_parts or claimed_parts)
    supporting_ids = _supporting_image_ids(vlm_output.images, claimed_parts, claimed_issues)
    risk_flags_text = ";".join(risk_flags) if risk_flags else "none"

    return FinalRow(
        user_id=context.user_id,
        image_paths=context.image_paths,
        user_claim=context.user_claim,
        claim_object=context.claim_object,
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=_evidence_reason(
            evidence_standard_met, all_parts_visible, valid_image, risk_flags
        ),
        risk_flags=risk_flags_text,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=_status_justification(
            claim_status, risk_flags, all_parts_visible, issues_match
        ),
        supporting_image_ids=";".join(supporting_ids),
        valid_image=valid_image,
        severity=vlm_output.overall_severity,
    )


def _assemble_risk_flags(
    context: ClaimContext,
    text_output: TextExtractorOutput,
    vlm_output: VLMInspectorOutput,
    claimed_parts: List[str],
    claimed_issues: List[str],
    visible_parts: List[str],
    visible_issues: List[str],
    extra_risk_flags: List[str] = [],
) -> List[str]:
    flags: List[str] = list(extra_risk_flags)
    for image in vlm_output.images:
        flags.extend(_clean_values(image.quality_flags, keep_none=False))
        if image.is_manipulated:
            flags.append("possible_manipulation")
        if image.visual_injection_detected:
            flags.append("text_instruction_present")
    if text_output.text_injection_detected:
        flags.append("text_instruction_present")
    if _history_is_risky(context.user_history):
        flags.extend(["user_history_risk", "manual_review_required"])
    if claimed_parts and claimed_parts != ["unknown"]:
        missing_parts = [part for part in claimed_parts if part not in visible_parts]
        if missing_parts and visible_parts:
            flags.append("wrong_object_part")
        elif missing_parts:
            flags.append("damage_not_visible")
    if not _issues_match(claimed_issues, visible_issues):
        flags.append("claim_mismatch")
    return _dedupe(flags) or ["none"]


def _history_is_risky(history: Optional[dict]) -> bool:
    if not history:
        return False
    for key, value in history.items():
        normalized = str(key).strip().lower()
        if normalized in HISTORY_REJECTED_FIELDS | HISTORY_MANUAL_FIELDS:
            try:
                if float(value) > 0:
                    return True
            except (TypeError, ValueError):
                if str(value).strip().lower() in {"true", "yes", "y"}:
                    return True
    return False


def _all_claimed_visible(claimed_parts: Sequence[str], visible_parts: Sequence[str]) -> bool:
    claimed = [part for part in claimed_parts if part not in {"unknown", "none"}]
    if not claimed:
        return bool(visible_parts)
    return all(part in visible_parts for part in claimed)


def _issues_match(claimed_issues: Sequence[str], visible_issues: Sequence[str]) -> bool:
    claimed = [issue for issue in claimed_issues if issue not in {"unknown"}]
    visible = [issue for issue in visible_issues if issue not in {"unknown"}]
    if not claimed:
        return True
    if not visible:
        return False
    if claimed == ["none"]:
        return visible == ["none"]
    if visible == ["none"]:
        return False
    return all(any(_compatible_issue(claim, seen) for seen in visible) for claim in claimed)


def _compatible_issue(claimed: str, visible: str) -> bool:
    if claimed == visible:
        return True
    compatible_groups = [
        {"broken_part", "missing_part"},
        {"crack", "glass_shatter"},
        {"dent", "crushed_packaging"},
    ]
    return any(claimed in group and visible in group for group in compatible_groups)


def _any_damage_visible(visible_issues: Sequence[str]) -> bool:
    return any(issue not in {"none", "unknown"} for issue in visible_issues)


def _supporting_image_ids(
    images: Sequence[ImageAnalysis],
    claimed_parts: Sequence[str],
    claimed_issues: Sequence[str],
) -> List[str]:
    supporting = []
    for image in images:
        part_ok = not claimed_parts or any(part in image.visible_parts for part in claimed_parts)
        issue_ok = not claimed_issues or any(
            _compatible_issue(claim, visible)
            for claim in claimed_issues
            for visible in image.visible_issues
        )
        if part_ok and issue_ok:
            supporting.append(image.image_id)
    return _dedupe(supporting)


def _evidence_reason(
    evidence_standard_met: bool,
    all_parts_visible: bool,
    valid_image: bool,
    risk_flags: Sequence[str],
) -> str:
    if evidence_standard_met:
        return "The claimed part or parts are visible and the submitted image quality is sufficient for review."
    if not valid_image:
        return "The image is invalid for automated review because it is cropped, manipulated, the wrong object, or the wrong part."
    if not all_parts_visible:
        return "The submitted evidence does not show every claimed part clearly enough for verification."
    blocking = [flag for flag in risk_flags if flag in QUALITY_BLOCKING_FLAGS]
    if blocking:
        return f"The evidence quality is insufficient due to: {';'.join(blocking)}."
    return "The submitted evidence is insufficient for automated verification."


def _status_justification(
    claim_status: ClaimStatus,
    risk_flags: Sequence[str],
    all_parts_visible: bool,
    issues_match: bool,
) -> str:
    if claim_status == ClaimStatus.CONTRADICTED:
        return "The visible evidence contradicts the claimed damage or severity."
    if claim_status == ClaimStatus.SUPPORTED:
        return "The visible evidence supports the claimed part and issue."
    if "damage_not_visible" in risk_flags or not all_parts_visible:
        return "The claimed damage or claimed part is not visible enough to verify."
    if not issues_match:
        return "The visible issue does not match the claimed issue."
    return "There is not enough reliable visual evidence to decide the claim."


def _clean_values(values: Iterable[Any], keep_none: bool = True) -> List[str]:
    cleaned = []
    for value in values or []:
        text = str(value).strip().lower()
        if not text or text == "nan":
            continue
        if text == "none" and not keep_none:
            continue
        cleaned.append(text)
    return _dedupe(cleaned)


def _flatten(groups: Iterable[Iterable[Any]]) -> List[Any]:
    flattened: List[Any] = []
    for group in groups:
        flattened.extend(group or [])
    return flattened


def _dedupe(values: Iterable[Any]) -> List[str]:
    ordered = OrderedDict()
    for value in values:
        text = str(value).strip()
        if text:
            ordered[text] = None
    return list(ordered.keys())


def _join_or_unknown(values: Sequence[str]) -> str:
    cleaned = [value for value in values if value]
    return ";".join(cleaned) if cleaned else IssueType.UNKNOWN.value

