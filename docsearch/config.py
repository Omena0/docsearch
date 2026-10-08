"""Configuration for docsearch."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Embedding model configuration."""

    name: str = "BAAI/bge-small-en-v1.5"
    device: str | None = None  # auto-detect
    batch_size: int = 128
    cache_dir: Path | None = None
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    def __post_init__(self):
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)


@dataclass
class ChunkingConfig:
    """Chunking configuration."""

    target_tokens: int = 400
    overlap_tokens: int = 50
    max_tokens: int = 512  # model max context


@dataclass
class IndexConfig:
    """Index storage configuration."""

    path: Path
    embedding_dim: int = 384  # bge-small-en-v1.5

    def __post_init__(self):
        self.path = Path(self.path)


@dataclass
class SearchConfig:
    """Search configuration."""

    top_k: int = 5
    top_files: int = 10
    top_nodes: int = 50
    lexical_boost: float = 0.1
    min_score: float = 0.0
    context_expansion: str = "parent_heading"  # parent_heading, full_ancestry, none


@dataclass
class Config:
    """Main configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    index: IndexConfig | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    ignore_dirs: tuple[str, ...] = (".git", "node_modules", "__pycache__", ".venv", "venv")
    markdown_extensions: tuple[str, ...] = (".md", ".markdown")

    def __post_init__(self):
        if self.index is not None and isinstance(self.index, dict):
            self.index = IndexConfig(**self.index)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.chunking, dict):
            self.chunking = ChunkingConfig(**self.chunking)
        if isinstance(self.search, dict):
            self.search = SearchConfig(**self.search)