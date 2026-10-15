"""Embedding cache indexed by content hash."""

import os
from pathlib import Path

import numpy as np

CACHE_DIR_NAME = "cache"


def get_cache_dir(index_path: Path) -> Path:
    """Get the cache directory for an index (inside the index directory)."""
    return Path(index_path) / CACHE_DIR_NAME


def get_cache_path(index_path: Path, content_hash: str) -> Path:
    """Get the cache file path for a content hash."""
    return get_cache_dir(index_path) / f"{content_hash}.npy"


def load_cached_embeddings(index_path: Path, content_hash: str):
    """Load cached embeddings for a content hash, if available."""
    cache_path = get_cache_path(index_path, content_hash)
    if cache_path.exists():
        return np.load(cache_path)
    return None


def save_cached_embeddings(index_path: Path, content_hash: str, embeddings: np.ndarray):
    """Save embeddings to cache, atomically."""
    cache_path = get_cache_path(index_path, content_hash)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    np.save(str(tmp_path) + ".npy", embeddings)
    os.replace(str(tmp_path) + ".npy", str(cache_path))