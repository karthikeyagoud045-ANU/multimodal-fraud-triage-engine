"""
security/exif_forensics.py — Local EXIF Metadata Forensics Module.

Analyses JPEG/TIFF image metadata to detect signs of fraud or image manipulation
WITHOUT any API calls. Runs entirely on the CPU using the `exifread` library.

WHY THIS MATTERS FOR FRAUD DETECTION:
--------------------------------------
Legitimate claim photos taken with a smartphone typically have:
  - A DateTimeOriginal tag (set by the camera at capture time).
  - A realistic capture date (not years in the past or future).
  - A Make/Model tag matching a real camera or phone manufacturer.
  - No suspicious Software tag (editing tools like Photoshop overwrite this).

Fraudulent images are often:
  - Screenshots (no EXIF metadata at all — stripped by the OS).
  - Images downloaded from the internet (date may be very old).
  - Photos edited in Photoshop/GIMP (Software tag reveals the editing tool).
  - Composited images where EXIF is entirely absent or inconsistent.

DESIGN PRINCIPLE — Conservative Flagging:
------------------------------------------
This module flags SUSPICIONS, not certainties. A single flag means "requires
closer human review" — not "reject automatically". The Rule Engine uses these
flags as inputs to risk scoring, not hard blockers.
"""
from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# If a photo was taken more than this many days ago, it may be a stock photo
# or a recycled image from a previous (possibly fraudulent) claim.
STALE_PHOTO_THRESHOLD_DAYS: int = 30

# These software tags are set by image editing applications.
# Their presence strongly suggests the image was post-processed.
SUSPICIOUS_SOFTWARE_KEYWORDS: tuple[str, ...] = (
    "photoshop",
    "lightroom",
    "gimp",
    "affinity",
    "snapseed",
    "facetune",
    "meitu",
    "picsart",
    "canva",
    "illustrator",
    "pixelmator",
    "paint.net",
    "corel",
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse_exif(image_path: Path) -> List[str]:
    """Analyse EXIF metadata of *image_path* and return a list of security flags.

    Args:
        image_path: Absolute or resolved path to the image file.
                    The file must exist before calling this function.

    Returns:
        A list of security flag strings. An empty list means the image passed
        all EXIF checks. Possible flags:
          - "possible_manipulation": Metadata is missing, stale, or shows editing.
          - "suspicious_software":   A known editing tool's name is in the EXIF.

        Note: Both flags may be returned simultaneously for a single image.

    Raises:
        Nothing — all exceptions are caught internally. If EXIF cannot be read
        (e.g., PNG without EXIF, corrupt file), the function returns [].
    """
    if not image_path.is_file():
        # File doesn't exist — the image processor will handle the missing-file
        # error separately. We don't double-flag here.
        return []

    try:
        import exifread  # type: ignore
    except ImportError:
        warnings.warn(
            "exifread is not installed. EXIF forensics will be skipped. "
            "Install it with: pip install exifread",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    flags: List[str] = []

    try:
        with open(image_path, "rb") as fh:
            # stop_tag limits parsing to what we actually need.
            # details=False skips MakerNote (speeds up parsing significantly).
            tags = exifread.process_file(fh, stop_tag="GPS GPSLatitude", details=False)
    except Exception:
        # Corrupt EXIF block, unsupported format, etc.
        # Treat as suspicious — legitimate unedited JPEGs almost always have EXIF.
        flags.append("possible_manipulation")
        return flags

    # ── Check 1: Capture Timestamp ────────────────────────────────────────────
    flags.extend(_check_capture_date(tags))

    # ── Check 2: Editing Software Tag ─────────────────────────────────────────
    flags.extend(_check_software_tag(tags))

    # ── Check 3: Implausible GPS Precision ────────────────────────────────────
    # Real GPS coordinates from phones have limited precision.
    # Artificially precise values (many decimal places) suggest synthetic metadata.
    flags.extend(_check_gps_precision(tags))

    # Deduplicate while preserving order
    seen: dict[str, None] = {}
    return [seen.setdefault(f, f) for f in flags if f not in seen]


def analyse_exif_from_path(path_str: str) -> List[str]:
    """Convenience wrapper that accepts a string path instead of a Path object.

    Args:
        path_str: String path to the image file.

    Returns:
        Same as analyse_exif().
    """
    return analyse_exif(Path(path_str))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_capture_date(tags: dict) -> List[str]:
    """Flag images with missing or suspiciously old capture timestamps.

    EXIF DateTimeOriginal is set at the moment of capture by the camera/phone.
    Its absence typically means the image was screenshotted, downloaded, or
    had its metadata stripped by an editing tool.
    """
    # Try primary and fallback date fields in priority order
    date_value: Optional[str] = None
    for tag_key in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        if tag_key in tags:
            date_value = str(tags[tag_key]).strip()
            break

    if date_value is None:
        # No capture date anywhere in the EXIF → strongly suspicious
        return ["possible_manipulation"]

    # Attempt to parse the standard EXIF date format: "YYYY:MM:DD HH:MM:SS"
    try:
        capture_dt = datetime.datetime.strptime(date_value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        # Malformed date string — not a standard camera format
        return ["possible_manipulation"]

    # Guard against implausible future dates (camera clock set far ahead)
    now = datetime.datetime.now()
    if capture_dt > now:
        return ["possible_manipulation"]

    # Flag photos older than the stale threshold
    age_days = (now - capture_dt).days
    if age_days > STALE_PHOTO_THRESHOLD_DAYS:
        return ["possible_manipulation"]

    return []


def _check_software_tag(tags: dict) -> List[str]:
    """Flag images whose EXIF Software tag reveals post-processing tools.

    Real unedited photos have a Software tag like "15.7.5" (iOS firmware version)
    or "CameraFirmware v2.1". Editing software like Photoshop overwrites this
    with its own name during Save/Export.
    """
    software_tag = tags.get("Image Software")
    if software_tag is None:
        return []  # Absent Software tag is normal and not suspicious

    software_str = str(software_tag).lower()
    for keyword in SUSPICIOUS_SOFTWARE_KEYWORDS:
        if keyword in software_str:
            return ["possible_manipulation", "suspicious_software"]

    return []


def _check_gps_precision(tags: dict) -> List[str]:
    """Flag images with implausibly precise GPS coordinates.

    Smartphone GPS has ~5-10 metre accuracy, which translates to about
    4-5 decimal places in degrees. Rational values with very large numerators
    (many significant figures) suggest the GPS data was synthesised by software.
    """
    lat_tag = tags.get("GPS GPSLatitude")
    if lat_tag is None:
        return []  # No GPS data is fine — most indoor photos lack it

    try:
        raw = str(lat_tag)  # e.g., "[51, 30, 12345678/1000000]"
        # Extract the seconds component (third element)
        parts = raw.strip("[]").split(", ")
        if len(parts) < 3:
            return []

        seconds_str = parts[2].strip()
        if "/" in seconds_str:
            numerator_str, denominator_str = seconds_str.split("/")
            numerator = int(numerator_str.strip())
            denominator = int(denominator_str.strip())
            if denominator == 0:
                return []
            # More than 6 significant figures in the seconds denominator is
            # extremely unusual for a real smartphone GPS reading.
            if len(str(denominator)) > 6:
                return ["possible_manipulation"]
    except Exception:
        # Any parsing failure is non-critical — skip this check
        pass

    return []
