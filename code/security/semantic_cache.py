"""
semantic_cache.py — Perceptual-hash-based duplicate image detector.

Maintains an in-memory registry of processed image hashes.
If a claim submits an image whose perceptual hash (phash) is identical
or near-identical (Hamming distance ≤ 4) to a **rejected** image already
seen in this pipeline run, it flags `non_original_image`.

This catches copy-paste fraud where a claimant resubmits the same photo
of damage they previously had rejected.

Uses `imagehash` (already in requirements). Zero API calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HAMMING_THRESHOLD = 4  # bits — two images within 4 bits are "the same"


@dataclass
class _CacheEntry:
    user_id: str
    claim_status: str  # populated after rule engine runs
    phash_str: str


class SemanticCache:
    """
    Singleton-safe in-process cache for one pipeline run.
    Register images as they are processed; flag duplicates of rejected ones.
    """

    def __init__(self) -> None:
        self._entries: List[_CacheEntry] = []

    def check(self, phash_str: Optional[str], image_path: str) -> List[str]:
        """
        Check *phash_str* against the cache of previously rejected images.

        Returns ["non_original_image"] if a near-duplicate rejected image
        is found, else [].
        """
        if not phash_str:
            return []

        try:
            import imagehash  # type: ignore

            query_hash = imagehash.hex_to_hash(phash_str)
            for entry in self._entries:
                if entry.claim_status not in {"rejected", "contradicted"}:
                    continue
                stored_hash = imagehash.hex_to_hash(entry.phash_str)
                distance = query_hash - stored_hash
                if distance <= HAMMING_THRESHOLD:
                    return ["non_original_image"]
        except Exception:
            pass

        return []

    def register(self, phash_str: Optional[str], user_id: str, claim_status: str = "pending") -> None:
        """Add or update an entry in the cache."""
        if not phash_str:
            return
        # Update existing entry if same phash
        for entry in self._entries:
            if entry.phash_str == phash_str:
                entry.claim_status = claim_status
                return
        self._entries.append(_CacheEntry(user_id=user_id, claim_status=claim_status, phash_str=phash_str))

    def update_status(self, phash_str: Optional[str], claim_status: str) -> None:
        """Call after rule engine resolves to mark the image's outcome."""
        if not phash_str:
            return
        for entry in self._entries:
            if entry.phash_str == phash_str:
                entry.claim_status = claim_status
                return


# Module-level singleton reused across all claims in a single pipeline run
_cache: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


def reset_cache() -> None:
    """Call between test runs to get a fresh cache."""
    global _cache
    _cache = None
