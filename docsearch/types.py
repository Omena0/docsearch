"""Type definitions for docsearch."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SemanticNode:
    """A semantic node in the Markdown hierarchy."""

    id: str
    file: str
    start_line: int
    end_line: int
    heading_level: int  # 0 for file-level, 1-6 for headings
    heading_text: str = ""
    heading_path: tuple[str, ...] = ()
    parent_id: str | None = None
    retrieval_text: str = ""
    source_text: str = ""
    embedding: np.ndarray | None = field(default=None, repr=False)
    content_hash: str = ""
    level: int = 0  # 0=file, 1=H1, 2=H2, etc.
    embedding_idx: int | None = None

    def __post_init__(self):
        if self.heading_path is None:
            self.heading_path = ()


@dataclass
class FileNode:
    """A file-level node with metadata."""

    id: str
    file: str
    start_line: int
    end_line: int
    heading_path: tuple[str, ...] = ()
    content_hash: str = ""
    embedding: np.ndarray | None = field(default=None, repr=False)
    child_count: int = 0
    embedding_idx: int | None = None


@dataclass
class SearchResult:
    """Search result with location and metadata."""

    file: str
    start_line: int
    end_line: int
    score: float
    heading_path: tuple[str, ...]
    matched_node_id: str
    matched_level: int
    matched_text: str = ""
    source_text: str = ""
    markdown_context: str = ""
    node: SemanticNode | None = None

    @property
    def location(self) -> str:
        """Return location string like 'file.md:10-20'."""
        return f"{self.file}:{self.start_line}-{self.end_line}"

    @property
    def heading_str(self) -> str:
        """Return heading path as string."""
        return " > ".join(self.heading_path)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "heading_path": list(self.heading_path),
            "matched_node_id": self.matched_node_id,
            "matched_level": self.matched_level,
            "matched_text": self.matched_text[:200] if self.matched_text else "",
            "location": self.location,
            "heading_str": self.heading_str,
        }


@dataclass
class IndexMetadata:
    """Index metadata."""

    version: int = 1
    model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    file_count: int = 0
    node_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    config_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "file_count": self.file_count,
            "node_count": self.node_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IndexMetadata":
        return cls(**data)