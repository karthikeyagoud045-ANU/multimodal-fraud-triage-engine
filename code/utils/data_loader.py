from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from models import ClaimContext, ClaimObject


OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def read_claims(claims_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(claims_csv)
    required = {"user_id", "image_paths", "user_claim", "claim_object"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required claims columns: {sorted(missing)}")
    return df


def read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def load_contexts(
    claims_csv: Path,
    user_history_csv: Optional[Path] = None,
    evidence_requirements_csv: Optional[Path] = None,
) -> List[ClaimContext]:
    claims = read_claims(claims_csv)
    base_dir = claims_csv.parent
    history = read_optional_csv(user_history_csv or base_dir / "user_history.csv")
    requirements = read_optional_csv(
        evidence_requirements_csv or base_dir / "evidence_requirements.csv"
    )

    contexts: List[ClaimContext] = []
    for row in claims.to_dict(orient="records"):
        user_id = str(row["user_id"])
        claim_object = ClaimObject(str(row["claim_object"]).strip().lower())
        contexts.append(
            ClaimContext(
                user_id=user_id,
                image_paths=str(row["image_paths"]),
                user_claim=str(row["user_claim"]),
                claim_object=claim_object,
                user_history=_row_for_user(history, user_id),
                evidence_requirements=_row_for_object(requirements, claim_object.value),
            )
        )
    return contexts


def write_output(rows: Iterable[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows), columns=OUTPUT_COLUMNS)
    df.to_csv(output_csv, index=False)


def split_image_paths(image_paths: str) -> List[str]:
    return [part.strip() for part in str(image_paths).split(";") if part.strip()]


def image_id_from_path(path: str) -> str:
    return Path(path).stem


def _row_for_user(df: Optional[pd.DataFrame], user_id: str) -> Optional[dict]:
    if df is None or "user_id" not in df.columns:
        return None
    match = df[df["user_id"].astype(str) == user_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _row_for_object(df: Optional[pd.DataFrame], claim_object: str) -> Optional[dict]:
    if df is None:
        return None
    for col in ("claim_object", "object_type", "object"):
        if col in df.columns:
            match = df[df[col].astype(str).str.lower() == claim_object]
            if not match.empty:
                return match.iloc[0].to_dict()
    return None

