"""Embedding model wrapper with BGE-small-en-v1.5."""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from docsearch.config import ModelConfig

# Suppress HF Hub unauthenticated warning
warnings.filterwarnings("ignore", message=".*unauthenticated.*", module="huggingface_hub.*")


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model: SentenceTransformer | None = None
        self._device: str | None = None

    @classmethod
    def get_instance(cls, config: ModelConfig | None = None) -> "EmbeddingModel":
        """Get singleton instance."""
        if cls._instance is None:
            if config is None:
                config = ModelConfig()
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def device(self) -> str:
        """Get device."""
        if self._device is None:
            self._device = self._detect_device()
        return self._device

    def _detect_device(self) -> str:
        """Detect best available device."""
        if self.config.device:
            return self.config.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        """Load the embedding model."""
        cache_dir = self.config.cache_dir
        if cache_dir is None:
            cache_dir = Path(os.environ.get("DOCSEARCH_CACHE", Path.home() / ".cache" / "docsearch"))

        self._model = SentenceTransformer(
            self.config.name,
            cache_folder=str(cache_dir),
            device=self.device,
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode documents (no instruction prefix)."""
        if not texts:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        batch_size = batch_size or self.config.batch_size
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode queries with BGE retrieval instruction."""
        if not texts:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        batch_size = batch_size or self.config.batch_size
        prefixed = [self.config.query_instruction + text for text in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        try:
            return self.model.get_embedding_dimension()
        except AttributeError:
            return self.model.get_sentence_embedding_dimension()


# Backward compatibility functions
def _get_model(config: ModelConfig | None = None) -> EmbeddingModel:
    return EmbeddingModel.get_instance(config)


def encode_documents(texts: list[str], batch_size: int = 128, config: ModelConfig | None = None) -> np.ndarray:
    """Encode documents for indexing."""
    model = _get_model(config)
    return model.encode_documents(texts, batch_size=batch_size)


def encode_queries(texts: list[str], batch_size: int = 128, config: ModelConfig | None = None) -> np.ndarray:
    """Encode queries for search."""
    model = _get_model(config)
    return model.encode_queries(texts, batch_size=batch_size)