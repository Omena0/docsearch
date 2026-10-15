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
    source_extensions: tuple[str, ...] = (
        ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
        ".cc", ".cxx", ".go", ".rs", ".java", ".kt", ".swift", ".rb",
        ".php", ".sh", ".bash", ".zsh", ".fish", ".sql", ".toml", ".yaml",
        ".yml", ".json", ".xml", ".html", ".css", ".scss", ".sass",
        ".lua", ".pl", ".pm", ".scala", ".clj", ".cljs", ".ex", ".exs",
        ".erl", ".hs", ".ml", ".fs", ".fsx", ".vim", ".dockerfile",
        ".makefile", ".mk", ".cmake", ".gradle", ".groovy", ".r",
        ".dart", ".vue", ".svelte", ".tex", ".bib", ".awk", ".tcl",
        ".s", ".asm", '.ld', 'makefile'
    )
    source_weight: float = 0.3  # Lower weight for source code vs docs
    auto_reindex: bool = True  # Auto re-index changed files on search
    ignored_globs: list[str] = field(default_factory=list)
    allowed_globs: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.index is not None and isinstance(self.index, dict):
            self.index = IndexConfig(**self.index)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.chunking, dict):
            self.chunking = ChunkingConfig(**self.chunking)
        if isinstance(self.search, dict):
            self.search = SearchConfig(**self.search)
        # Ensure paths are Path objects
        if self.model.cache_dir is not None:
            self.model.cache_dir = Path(self.model.cache_dir)
        if self.index is not None:
            self.index.path = Path(self.index.path)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": {
                "name": self.model.name,
                "device": self.model.device,
                "batch_size": self.model.batch_size,
                "cache_dir": str(self.model.cache_dir) if self.model.cache_dir else None,
                "query_instruction": self.model.query_instruction,
            },
            "chunking": {
                "target_tokens": self.chunking.target_tokens,
                "overlap_tokens": self.chunking.overlap_tokens,
                "max_tokens": self.chunking.max_tokens,
            },
            "search": {
                "top_k": self.search.top_k,
                "top_files": self.search.top_files,
                "top_nodes": self.search.top_nodes,
                "lexical_boost": self.search.lexical_boost,
                "min_score": self.search.min_score,
                "context_expansion": self.search.context_expansion,
            },
            "ignore_dirs": list(self.ignore_dirs),
            "markdown_extensions": list(self.markdown_extensions),
            "source_extensions": list(self.source_extensions),
            "source_weight": self.source_weight,
            "auto_reindex": self.auto_reindex,
            "ignored_globs": self.ignored_globs,
            "allowed_globs": self.allowed_globs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary."""
        model = ModelConfig(**data.get("model", {}))
        chunking = ChunkingConfig(**data.get("chunking", {}))
        search = SearchConfig(**data.get("search", {}))
        index = None
        if "index" in data:
            index = IndexConfig(**data["index"])
        default_dirs = [".git", "node_modules", "__pycache__", ".venv", "venv"]
        default_exts = (
            ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
            ".cc", ".cxx", ".go", ".rs", ".java", ".kt", ".swift", ".rb",
        )
        return cls(
            model=model,
            chunking=chunking,
            index=index,
            search=search,
            ignore_dirs=tuple(data.get("ignore_dirs", default_dirs)),
            markdown_extensions=tuple(data.get("markdown_extensions", (".md", ".markdown"))),
            source_extensions=tuple(data.get("source_extensions", default_exts)),
            source_weight=data.get("source_weight", 0.3),
            auto_reindex=data.get("auto_reindex", True),
            ignored_globs=data.get("ignored_globs", []),
            allowed_globs=data.get("allowed_globs", []),
        )
