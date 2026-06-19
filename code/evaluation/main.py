"""
evaluation/main.py — Hackathon Evaluation & Performance Report Generator.

WHAT THIS SCRIPT DOES:
-----------------------
1. Compares output.csv (pipeline predictions) against sample_claims.csv (ground truth).
2. Computes per-column accuracy for the key adjudication fields.
3. Reads the operational metrics from output.metrics.json.
4. Generates a structured evaluation_report.md with:
   - Accuracy scores (claim_status, issue_type, object_part)
   - Operational analysis (latency, token usage, cost)
   - Architecture justification (why Cascade > Monolithic)
   - Security posture summary

USAGE:
------
    # After running main.py on sample_claims.csv:
    python3 evaluation/main.py \\
        --pred ../../dataset/sample_output.csv \\
        --truth ../../dataset/sample_claims.csv \\
        --metrics ../../dataset/sample_output.metrics.json \\
        --report evaluation/evaluation_report.md

COLUMN MATCHING NOTES:
----------------------
The evaluation compares predictions on user_id (inner join).
Multi-value fields (e.g., "scratch;dent") are compared as:
  - Exact string match (primary metric)
  - Set overlap accuracy (secondary metric — more lenient)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Try canonical hackathon layout first, then fallback options
_DATASET_CANDIDATES = [
    PROJECT_ROOT.parent / "claims",
    PROJECT_ROOT.parent / "dataset",
    PROJECT_ROOT / "dataset",
]
DATA_ROOT = next((p for p in _DATASET_CANDIDATES if p.exists()), _DATASET_CANDIDATES[0])
EVAL_DIR  = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Columns Evaluated
# ─────────────────────────────────────────────────────────────────────────────

# Primary columns: exact string match required
PRIMARY_COLUMNS = ["claim_status", "issue_type", "object_part"]

# Secondary columns: compared where ground-truth values exist
SECONDARY_COLUMNS = ["valid_image", "severity", "evidence_standard_met", "risk_flags"]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    prediction_csv: Path,
    truth_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compare prediction CSV against ground truth and compute accuracy metrics.

    Args:
        prediction_csv: Path to output.csv produced by main.py.
        truth_csv:      Path to sample_claims.csv (ground truth with labels).

    Returns:
        (merged_df, metrics_dict) where:
          - merged_df has _pred and _truth columns for every evaluated field.
          - metrics_dict has accuracy scores for each column.

    Raises:
        ValueError: If either CSV is empty or there are no overlapping user_ids.
    """
    pred  = pd.read_csv(prediction_csv, dtype=str).fillna("")
    truth = pd.read_csv(truth_csv,      dtype=str).fillna("")

    if pred.empty:
        raise ValueError(f"Prediction file is empty: {prediction_csv}")
    if truth.empty:
        raise ValueError(f"Ground truth file is empty: {truth_csv}")

    # Inner join on user_id — only evaluate rows present in both files
    merged = pred.merge(truth, on="user_id", suffixes=("_pred", "_truth"))
    if merged.empty:
        raise ValueError(
            "No overlapping user_id values between prediction and truth files.\n"
            f"Prediction user_ids: {pred['user_id'].tolist()[:5]}\n"
            f"Truth user_ids:      {truth['user_id'].tolist()[:5]}"
        )

    metrics: Dict[str, float] = {"rows_compared": float(len(merged))}

    evaluated_columns: List[str] = []

    for col in PRIMARY_COLUMNS + SECONDARY_COLUMNS:
        pred_col  = f"{col}_pred"
        truth_col = f"{col}_truth"
        if pred_col not in merged.columns or truth_col not in merged.columns:
            continue  # Column missing from one file — skip

        # ── Exact string match ──────────────────────────────────────────────
        exact = merged[pred_col].str.strip() == merged[truth_col].str.strip()
        metrics[f"{col}_exact_accuracy"] = float(exact.mean())

        # ── Set overlap accuracy (for semicolon-delimited multi-values) ────
        # E.g., pred="scratch;dent" vs truth="dent;scratch" → 1.0
        # E.g., pred="scratch" vs truth="dent;scratch" → 0.5 (partial credit)
        set_scores = []
        for p_val, t_val in zip(merged[pred_col], merged[truth_col]):
            p_set = {v.strip() for v in str(p_val).split(";") if v.strip()}
            t_set = {v.strip() for v in str(t_val).split(";") if v.strip()}
            if not t_set:
                set_scores.append(1.0 if not p_set else 0.0)
            else:
                overlap = len(p_set & t_set)
                union   = len(p_set | t_set)
                set_scores.append(overlap / union if union else 0.0)
        metrics[f"{col}_set_accuracy"] = round(
            sum(set_scores) / len(set_scores), 4
        ) if set_scores else 0.0

        evaluated_columns.append(col)

    # ── Overall row exact match ─────────────────────────────────────────────
    # A row is "correct" only if ALL primary columns match exactly.
    if all(f"{c}_pred" in merged.columns and f"{c}_truth" in merged.columns
           for c in PRIMARY_COLUMNS):
        row_exact = pd.Series(True, index=merged.index)
        for col in PRIMARY_COLUMNS:
            row_exact &= (
                merged[f"{col}_pred"].str.strip() == merged[f"{col}_truth"].str.strip()
            )
        metrics["row_exact_match_all_primary"] = float(row_exact.mean())

    # ── Per-status accuracy (breakdown within claim_status) ────────────────
    if "claim_status_pred" in merged.columns and "claim_status_truth" in merged.columns:
        for status in merged["claim_status_truth"].unique():
            mask = merged["claim_status_truth"].str.strip() == status
            sub = merged[mask]
            if len(sub) > 0:
                acc = (
                    sub["claim_status_pred"].str.strip() == sub["claim_status_truth"].str.strip()
                ).mean()
                metrics[f"claim_status_{status}_accuracy"] = float(acc)

    return merged, metrics


def _confusion_matrix_str(merged: pd.DataFrame) -> str:
    """Build a text-format confusion matrix for claim_status."""
    if "claim_status_pred" not in merged.columns or "claim_status_truth" not in merged.columns:
        return ""
    statuses = sorted(merged["claim_status_truth"].unique())
    header = "Predicted →       " + "  ".join(f"{s:22}" for s in statuses)
    lines = [header, "─" * len(header)]
    for truth_status in statuses:
        row_label = f"Actual {truth_status:16}"
        counts = []
        for pred_status in statuses:
            n = len(merged[
                (merged["claim_status_truth"].str.strip() == truth_status) &
                (merged["claim_status_pred"].str.strip() == pred_status)
            ])
            counts.append(f"{n:22}")
        lines.append(row_label + "  ".join(counts))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    metrics: Dict[str, float],
    merged: pd.DataFrame,
    operational_metrics: Optional[Dict],
    output_path: Path,
    pred_path: Path,
    truth_path: Path,
) -> str:
    """Write evaluation_report.md with accuracy, operational analysis, and strategy.

    Args:
        metrics:             Output from evaluate().
        merged:              Merged DataFrame from evaluate().
        operational_metrics: Dict from output.metrics.json (may be None).
        output_path:         Where to write the report.
        pred_path:           Path to prediction CSV (for display).
        truth_path:          Path to truth CSV (for display).

    Returns:
        The full report as a string.
    """
    rows = int(metrics.get("rows_compared", 0))
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def pct(key: str) -> str:
        v = metrics.get(key)
        return f"{v*100:.1f}%" if v is not None else "N/A"

    # ── Operational data extraction ────────────────────────────────────────────
    op   = operational_metrics or {}
    run  = op.get("run", {})
    tok  = op.get("token_estimates", {})
    tele = op.get("telemetry", {})
    lat  = tele.get("latency", {})
    mcalls = tele.get("model_calls", {})
    sec  = tele.get("security", {})
    errs = tele.get("errors", {})

    elapsed     = run.get("elapsed_seconds", "N/A")
    avg_s       = run.get("avg_seconds_per_claim", "N/A")
    concurrency = run.get("concurrency", 3)
    used_ai     = run.get("used_ai", True)
    used_sec    = run.get("used_security", True)

    confusion = _confusion_matrix_str(merged)

    # ── Compose report ─────────────────────────────────────────────────────────
    lines = [
        "# Evaluation Report — Multi-Modal Fraud Triage Engine",
        "",
        f"> **Generated:** {now}  |  **Compared:** `{pred_path.name}` vs `{truth_path.name}`  |  **Rows:** {rows}",
        "",
        "---",
        "",
        "## 1. Accuracy Results",
        "",
        "### Primary Metrics (Exact String Match)",
        "",
        "| Field | Exact Match | Set Overlap |",
        "|---|---|---|",
        f"| `claim_status` | **{pct('claim_status_exact_accuracy')}** | {pct('claim_status_set_accuracy')} |",
        f"| `issue_type`   | **{pct('issue_type_exact_accuracy')}** | {pct('issue_type_set_accuracy')} |",
        f"| `object_part`  | **{pct('object_part_exact_accuracy')}** | {pct('object_part_set_accuracy')} |",
        "",
        "### Composite",
        "",
        f"- **Row exact match (all primary columns):** {pct('row_exact_match_all_primary')}",
        "",
        "### Secondary Metrics",
        "",
        "| Field | Exact Match |",
        "|---|---|",
        f"| `valid_image` | {pct('valid_image_exact_accuracy')} |",
        f"| `severity`    | {pct('severity_exact_accuracy')} |",
        f"| `evidence_standard_met` | {pct('evidence_standard_met_exact_accuracy')} |",
        "",
    ]

    # Per-status breakdown
    status_keys = [k for k in metrics if k.startswith("claim_status_") and k.endswith("_accuracy") and "exact" not in k and "set" not in k]
    if status_keys:
        lines += [
            "### Claim Status Breakdown",
            "",
            "| Status | Recall |",
            "|---|---|",
        ]
        for k in sorted(status_keys):
            status = k.replace("claim_status_", "").replace("_accuracy", "")
            lines.append(f"| `{status}` | {pct(k)} |")
        lines.append("")

    # Confusion matrix
    if confusion:
        lines += [
            "### Confusion Matrix (claim_status)",
            "",
            "```",
            confusion,
            "```",
            "",
        ]

    # ── Operational Analysis ───────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 2. Operational Analysis",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total processing time | `{elapsed}s` |",
        f"| Avg time per claim | `{avg_s}s` |",
        f"| Concurrency level | `{concurrency}` |",
        f"| AI agents enabled | `{used_ai}` |",
        f"| Security pre-flight | `{used_sec}` |",
        f"| P90 latency | `{lat.get('p90_ms', 'N/A')}ms` |",
        f"| Max latency | `{lat.get('max_ms', 'N/A')}ms` |",
        "",
        "### Model Call Statistics",
        "",
        "| Provider | Model | Calls | Est. Tokens |",
        "|---|---|---|---|",
        f"| Groq | Llama-3.3-70B + 8B | {mcalls.get('groq_total', 'N/A')} | {f'{tok.get('groq_text_tokens', 0):,}' if isinstance(tok.get('groq_text_tokens'), (int, float)) else 'N/A'} |",
        f"| Nvidia NIM | Llama-3.2-90B Vision | {mcalls.get('nvidia_total', 'N/A')} | {f'{tok.get('nvidia_vision_tokens', 0):,}' if isinstance(tok.get('nvidia_vision_tokens'), (int, float)) else 'N/A'} |",
        f"| — | **Total** | — | **{f'{tok.get('total_estimated', 0):,}' if isinstance(tok.get('total_estimated'), (int, float)) else 'N/A'}** |",
        "",
        "### Cost Analysis",
        "",
        "> **Estimated inference cost: $0.00**",
        ">",
        "> Both Groq and Nvidia NIM are used at their free tier for this submission.",
        "> Equivalent OpenAI GPT-4o cost at `$0.005/1K tokens` would be approximately",
        f"> `${(tok.get('total_estimated', 0) / 1000) * 0.005:.2f}` — representing **100% savings**.",
        "",
    ]

    # Security stats
    lines += [
        "### Security Pre-Flight",
        "",
        "| Check | Count |",
        "|---|---|",
        f"| Total security flags raised | {sec.get('flags_raised', 0)} |",
        f"| Claims with VLM skipped (duplicate image) | {sec.get('vlm_skipped_count', 0)} |",
        f"| Failed model calls (recovered by tenacity) | {mcalls.get('failed_total', 0)} |",
        f"| Claims that required error fallback | {errs.get('count', 0)} |",
        "",
    ]

    # Flag frequency table
    flag_freq = sec.get("flag_frequency", {})
    if flag_freq:
        lines += [
            "#### Security Flag Frequency",
            "",
            "| Flag | Count |",
            "|---|---|",
        ]
        for flag, count in sorted(flag_freq.items(), key=lambda x: -x[1]):
            lines.append(f"| `{flag}` | {count} |")
        lines.append("")

    # ── Architecture: Cascade vs Monolithic ───────────────────────────────────
    lines += [
        "---",
        "",
        "## 3. Architecture: Why Cascade > Monolithic VLM",
        "",
        "Most hackathon submissions feed raw image + text directly into a single",
        "proprietary black-box (GPT-4o) and accept whatever JSON it returns. We",
        "deliberately chose a **Cascade Architecture** with three tiers:",
        "",
        "```",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│  TIER 1 — LOCAL SECURITY (Zero tokens, ~5ms)                        │",
        "│  EXIF Forensics → OCR Injection Scan → Semantic phash Cache         │",
        "│                                                                      │",
        "│  TIER 2 — OPEN-WEIGHTS AI (Groq + Nvidia, ~3-8s)                    │",
        "│  Llama-3.3-70B (text) + Llama-3.2-90B Vision (images)               │",
        "│                                                                      │",
        "│  TIER 3 — DETERMINISTIC LOGIC (Zero tokens, ~0ms)                   │",
        "│  Rule Engine + Risk Assessor → Guaranteed schema compliance          │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "```",
        "",
        "| Dimension | Cascade (Ours) | Monolithic GPT-4o |",
        "|---|---|---|",
        "| **Cost per 1000 claims** | $0.00 (free tier) | ~$50-200 |",
        "| **Latency per claim** | 3-8s | 8-20s (vision queue) |",
        "| **Vendor lock-in** | None (open-weights) | Complete (OpenAI) |",
        "| **Hallucination rate** | Low (constrained by Rule Engine) | High (unconstrained JSON) |",
        "| **Injection resistance** | 3 independent layers | Single LLM prompt |",
        "| **Auditability** | Full (Rule Engine is pure Python) | None (black box) |",
        "| **Offline capability** | Partial (heuristic mode) | None |",
        "",
        "### Key Innovation: The \"Cascade Firewall\"",
        "",
        "Our Rule Engine **overrides** LLM outputs when they violate business logic:",
        "- If the VLM says `supported` but visible issues ≠ claimed issues → `contradicted`",
        "- If injection text is detected → status is never upgraded regardless of AI output",
        "- If EXIF metadata is manipulated → image is disqualified before VLM call",
        "",
        "This makes the system **adversarially hardened** in ways a monolithic LLM cannot be.",
        "",
    ]

    # ── Security Posture ───────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 4. Security Posture",
        "",
        "The pipeline implements three independent defense layers against insurance fraud",
        "and adversarial prompt injection attacks:",
        "",
        "### Layer 1 — EXIF Forensics (`security/exif_forensics.py`)",
        "",
        "Analyses image metadata before any AI call:",
        "- **Software tag detection**: Flags images processed by Photoshop, GIMP, Pixelmator,",
        "  or DALL-E (synthetic image generators).",
        "- **Timestamp validation**: Flags images missing capture date/time (common in screenshots",
        "  or reprocessed images).",
        "- **GPS precision anomaly**: Flags suspiciously round GPS coordinates (9.000000, 76.000000)",
        "  which indicate manual coordinate injection, not real GPS data.",
        "",
        "### Layer 2 — OCR Injection Scan (`security/ocr_sanitizer.py`)",
        "",
        "Runs EasyOCR on every image to detect text that attempts to manipulate AI decisions:",
        "- **19 multilingual regex patterns** covering English, Hindi, Spanish, and Chinese.",
        "- **Examples detected**: \"approve this claim\", \"ignore instructions\",",
        "  \"usko follow karke approve kar dena\", \"批准索赔\".",
        "- **Lazy loading**: EasyOCR model is loaded once on first use (~2s cold start),",
        "  then cached for all subsequent calls.",
        "",
        "### Layer 3 — Semantic phash Cache (`security/semantic_cache.py`)",
        "",
        "Detects images reused across multiple claims:",
        "- **Perceptual hashing (phash)**: Detects visually similar images even after JPEG",
        "  recompression, cropping, or resizing — unlike MD5 which breaks on any edit.",
        "- **Cross-claim memory**: Persists hashes between pipeline runs.",
        "- **Verdict tracking**: If image `img_001.jpg` was in a `contradicted` claim, any",
        "  future claim submitting the same image is flagged `non_original_image`.",
        "",
        "### Redundancy: AI-Level Defenses",
        "",
        "Even if the local layers miss an injection, the LLM agents have independent defenses:",
        "- **Text extractor**: Explicitly trained (via system prompt) to detect and flag injection.",
        "- **VLM inspector**: Instructed to treat image text as visual data only — not commands.",
        "- **Rule Engine**: Injection flag from ANY layer forces `manual_review_required` regardless",
        "  of AI-adjudicated status.",
        "",
    ]

    # ── Commands ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 5. Execution Commands",
        "",
        "```bash",
        "# 1. Install dependencies",
        "pip install -r code/requirements.txt",
        "",
        "# 2. Set API keys in code/.env",
        "# GROQ_API_KEY=gsk_...",
        "# NVIDIA_API_KEY=nvapi-...",
        "",
        "# 3a. Test on sample_claims.csv (20 rows, ground truth available)",
        "python3 code/main.py \\",
        "    --claims-csv dataset/sample_claims.csv \\",
        "    --dataset-root dataset \\",
        "    --output-csv dataset/sample_output.csv \\",
        "    --user-history-csv dataset/user_history.csv \\",
        "    --evidence-requirements-csv dataset/evidence_requirements.csv",
        "",
        "# 3b. Full submission run on claims.csv",
        "python3 code/main.py \\",
        "    --claims-csv dataset/claims.csv \\",
        "    --dataset-root dataset \\",
        "    --output-csv dataset/output.csv \\",
        "    --user-history-csv dataset/user_history.csv \\",
        "    --evidence-requirements-csv dataset/evidence_requirements.csv",
        "",
        "# 3c. Heuristic dry-run (no API keys needed)",
        "python3 code/main.py --no-ai --no-security \\",
        "    --claims-csv dataset/sample_claims.csv \\",
        "    --output-csv dataset/sample_output_dryrun.csv",
        "",
        "# 4. Run evaluation report",
        "python3 code/evaluation/main.py \\",
        "    --pred dataset/sample_output.csv \\",
        "    --truth dataset/sample_claims.csv \\",
        "    --metrics dataset/sample_output.metrics.json \\",
        "    --report code/evaluation/evaluation_report.md",
        "```",
        "",
    ]

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline output against ground truth and generate report.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=DATA_ROOT / "output_test.csv",
        help="Prediction CSV (output of main.py).",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=DATA_ROOT / "sample_claims.csv",
        help="Ground truth CSV (sample_claims.csv with labelled columns).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Path to output.metrics.json generated by main.py (optional).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_DIR / "evaluation_report.md",
        help="Output path for evaluation_report.md.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Print metrics to stdout only; do not write a markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_path  = args.pred.expanduser().resolve()
    truth_path = args.truth.expanduser().resolve()

    print(f"\n📋 Evaluating: {pred_path.name}  vs  {truth_path.name}")

    merged, metrics = evaluate(pred_path, truth_path)

    # Load operational metrics if available
    op_metrics: Optional[Dict] = None
    metrics_path = args.metrics
    if metrics_path is None:
        # Try to auto-discover alongside the prediction CSV
        candidate = pred_path.with_suffix(".metrics.json")
        if candidate.exists():
            metrics_path = candidate
    if metrics_path and metrics_path.exists():
        try:
            op_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️  Could not load metrics JSON: {e}")

    # ── Print summary to stdout ────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  ACCURACY RESULTS")
    print(f"{'═'*50}")
    primary_keys = ["claim_status_exact_accuracy", "issue_type_exact_accuracy", "object_part_exact_accuracy"]
    for key in primary_keys:
        label = key.replace("_exact_accuracy", "").replace("_", " ").title()
        val   = metrics.get(key)
        bar   = "█" * int((val or 0) * 20) if val is not None else ""
        print(f"  {label:20} {(val or 0)*100:6.1f}%  {bar}")
    print(f"{'─'*50}")
    print(f"  Row exact match:     {metrics.get('row_exact_match_all_primary', 0)*100:.1f}%")
    print(f"  Rows compared:       {int(metrics.get('rows_compared', 0))}")
    print(f"{'═'*50}\n")

    if not args.no_report:
        report_path = args.report.expanduser().resolve()
        report = generate_report(
            metrics=metrics,
            merged=merged,
            operational_metrics=op_metrics,
            output_path=report_path,
            pred_path=pred_path,
            truth_path=truth_path,
        )
        print(f"📄 Report written → {report_path}")
    else:
        # Just print all metrics
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
