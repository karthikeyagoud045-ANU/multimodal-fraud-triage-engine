"""
utils/image_processor.py — Image Preprocessing Pipeline for VLM Token Efficiency.

Handles every image-related operation needed before sending images to Nvidia NIM:
  1. PATH RESOLUTION  — Tries multiple candidate locations for relative paths.
  2. LOADING & FIXING — Opens the image, corrects EXIF rotation, converts to RGB.
  3. RESIZING         — Downscales to max 1024px on the longest side (saves ~60% tokens).
  4. PERCEPTUAL HASH  — Computes a 64-bit phash for duplicate/fraud detection.
  5. BASE64 ENCODING  — Converts to a data URL for the VLM API payload.

WHY RESIZE TO 1024px?
----------------------
Nvidia NIM charges per image token. A 4000x3000px photo uses ~9x more tokens
than a 1024x768px version of the same image. For damage assessment tasks
(identifying dents, scratches, cracks), 1024px retains all the detail we need.
This alone reduces image-related API costs by ~60-80%.

WHY JPEG RE-ENCODE?
--------------------
Even if the original is a PNG, we re-encode as JPEG (quality=86) because:
  - JPEG at q86 is visually near-lossless for photographic content.
  - PNG can be 5-10x larger than JPEG for the same image content.
  - Smaller payload = fewer tokens = faster API response = lower cost.

WHY PERCEPTUAL HASH?
---------------------
Unlike a cryptographic hash (MD5/SHA), a perceptual hash (phash) is
similar for visually similar images even if they differ in size, compression,
or minor edits. This lets the semantic_cache.py detect:
  - Exact duplicates (same image, re-uploaded).
  - Near-duplicates (same image, slightly cropped or compressed differently).
  - Recycled fraud images (photo from a previous claim, resubmitted).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedImage:
    """All derived attributes of a single claim image after preprocessing.

    This dataclass is passed through the entire pipeline — from image_processor
    to security modules to the VLM agent.

    Attributes:
        image_id:      Filename stem (e.g., 'img_1' from 'img_1.jpg').
                       Used as the primary key in ImageAnalysis.image_id.
        original_path: The raw path string from the CSV (may be relative).
        resolved_path: Absolute Path object if the file was found, else None.
        data_url:      Base64-encoded JPEG data URL for the VLM API payload.
                       Format: "data:image/jpeg;base64,<encoded>".
                       None if the image could not be processed.
        phash:         64-character hex perceptual hash string.
                       None if imagehash is not installed or hashing failed.
        error:         Human-readable error message if processing failed.
                       None on success. When set, data_url and phash are None.
    """
    image_id: str
    original_path: str
    resolved_path: Optional[Path]
    data_url: Optional[str]
    phash: Optional[str]
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """True if the image was successfully processed and has a data URL."""
        return self.data_url is not None and self.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Path Resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_image_path(path_text: str, dataset_root: Path) -> Optional[Path]:
    """Resolve a relative or absolute image path to an existing file.

    The CSV stores relative paths like "images/sample/case_001/img_1.jpg".
    We try multiple candidate locations to handle different working directories
    and zip-extraction layouts.

    Resolution order:
        1. Treat as absolute path (if it happens to be absolute and exists).
        2. Relative to dataset_root (the directory containing the claims CSV).
        3. Relative to dataset_root's parent (one level up).
        4. Relative to the current working directory.

    Args:
        path_text:    Raw path string from the CSV.
        dataset_root: Directory containing the claims CSV file.

    Returns:
        An existing Path object if found, otherwise None.
    """
    # Fast path: absolute path that exists as-is
    candidate = Path(path_text)
    if candidate.is_absolute() and candidate.is_file():
        return candidate

    # Try each candidate location in priority order
    candidates: list[Path] = [
        dataset_root / candidate,          # Most common: relative to CSV dir
        dataset_root.parent / candidate,   # Relative to parent of CSV dir
        Path.cwd() / candidate,            # Relative to CWD (for debugging)
    ]

    for path in candidates:
        if path.is_file():
            return path.resolve()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single Image Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    path_text: str,
    dataset_root: Path,
    max_size: int = 1024,
    jpeg_quality: int = 86,
) -> ProcessedImage:
    """Load, resize, hash, and base64-encode a single claim image.

    Args:
        path_text:    Raw image path string from the CSV row.
        dataset_root: Root directory for resolving relative paths.
        max_size:     Maximum pixel dimension after resizing.
                      Both width and height will be ≤ max_size.
                      Aspect ratio is preserved (thumbnail mode).
        jpeg_quality: JPEG re-encode quality (50-100). 86 is near-lossless.

    Returns:
        ProcessedImage with all fields populated on success.
        ProcessedImage with error set and data_url=None on failure.
    """
    from utils.data_loader import image_id_from_path

    image_id = image_id_from_path(path_text)

    # ── Step 1: Resolve the file path ─────────────────────────────────────────
    resolved = resolve_image_path(path_text, dataset_root)
    if resolved is None:
        return ProcessedImage(
            image_id=image_id,
            original_path=path_text,
            resolved_path=None,
            data_url=None,
            phash=None,
            error=f"Image file not found: {path_text!r}. "
                  f"Searched in {dataset_root} and parent directories.",
        )

    # ── Step 2: Load, fix rotation, convert, resize ───────────────────────────
    try:
        from PIL import Image, ImageOps  # type: ignore

        with Image.open(resolved) as img:
            # EXIF transpose: corrects rotation metadata embedded by smartphones.
            # Without this, portrait photos may appear sideways to the VLM.
            img = ImageOps.exif_transpose(img)

            # Convert to RGB: handles RGBA (PNG with transparency), L (grayscale),
            # CMYK (some old JPEGs), and palette-mode (GIF) images.
            # The VLM expects standard 3-channel RGB.
            img = img.convert("RGB")

            # Resize using thumbnail(): preserves aspect ratio, never upscales.
            # A 4000x3000 image becomes 1024x768. A 500x500 image stays 500x500.
            img.thumbnail((max_size, max_size), Image.LANCZOS)

            # ── Step 3: Compute perceptual hash BEFORE re-encoding ────────────
            # We hash the PIL Image object (not the JPEG bytes) for consistency
            # across different compression levels.
            phash = _compute_perceptual_hash(img)

            # ── Step 4: Encode as JPEG data URL ──────────────────────────────
            buffer = BytesIO()
            img.save(
                buffer,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,      # Huffman-code optimisation (slightly slower, smaller file)
                progressive=False,  # Sequential JPEG works better with base64 APIs
            )
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            data_url = f"data:image/jpeg;base64,{encoded}"

        return ProcessedImage(
            image_id=image_id,
            original_path=path_text,
            resolved_path=resolved,
            data_url=data_url,
            phash=phash,
        )

    except Exception as exc:
        # Catch all PIL errors: corrupt files, unsupported formats, OOM, etc.
        return ProcessedImage(
            image_id=image_id,
            original_path=path_text,
            resolved_path=resolved,
            data_url=None,
            phash=None,
            error=f"Failed to process image {resolved.name}: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_images(
    paths: Iterable[str],
    dataset_root: Path,
    max_size: int = 1024,
    jpeg_quality: int = 86,
) -> List[ProcessedImage]:
    """Process a collection of image paths for a single claim.

    Args:
        paths:        Iterable of raw path strings (from split_image_paths).
        dataset_root: Root directory for resolving relative paths.
        max_size:     Maximum pixel dimension after resizing.
        jpeg_quality: JPEG re-encode quality.

    Returns:
        List of ProcessedImage objects, one per path.
        Failed images have error set and data_url=None — they are not skipped.
        The VLM agent and security modules check .is_valid before using an image.
    """
    return [
        process_image(path, dataset_root, max_size=max_size, jpeg_quality=jpeg_quality)
        for path in paths
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_perceptual_hash(image: "Image.Image") -> Optional[str]:  # type: ignore[name-defined]
    """Compute a 64-bit perceptual hash (phash) of a PIL Image.

    Perceptual hashing works by:
        1. Resizing the image to 32x32 pixels.
        2. Converting to grayscale.
        3. Computing the 2D DCT (discrete cosine transform).
        4. Comparing DCT coefficients to their mean to produce a 64-bit hash.

    Two images that are visually similar (different compression, slight crop,
    minor colour adjustment) will have hashes with a Hamming distance < 8.
    Completely different images have distances close to 32.

    Args:
        image: A PIL Image object (must be RGB or grayscale).

    Returns:
        64-character lowercase hex string representing the phash.
        None if imagehash is not installed or hashing raises an exception.
    """
    try:
        import imagehash  # type: ignore
        # phash() is the most robust algorithm for near-duplicate detection.
        # ahash() (average hash) is faster but more sensitive to minor changes.
        # dhash() (difference hash) is good for detecting cropping.
        return str(imagehash.phash(image))
    except ImportError:
        # imagehash not installed — duplicate detection will be disabled.
        # Not critical enough to warn on every image; warn once at startup instead.
        return None
    except Exception:
        # Image hashing can fail on very small or unusual images.
        return None
