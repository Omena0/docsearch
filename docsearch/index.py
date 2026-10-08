"""Main indexing logic."""

import hashlib
import time
from pathlib import Path

import numpy as np

from docsearch.chunking import chunk_document, hash_content
from docsearch.config import Config, IndexConfig
from docsearch.markdown import parse_markdown
from docsearch.model import EmbeddingModel
from docsearch.search import search
from docsearch.storage import (
    delete_file_nodes,
    get_all_files,
    get_connection,
    get_file_count,
    get_file_info,
    get_node_count,
    init_database,
    load_embeddings,
    load_file_nodes,
    load_metadata,
    load_semantic_nodes,
    save_embeddings,
    save_file_node,
    save_metadata,
    save_semantic_node,
)
from docsearch.types import FileNode, IndexMetadata, SearchResult, SemanticNode

SCHEMA_VERSION = 1

class DocSearchIndex:
    """Main entry point for the docsearch library."""

    def __init__(self, config: Config):
        self.config = config
        self._embeddings: np.ndarray | None = None
        self._file_nodes: list[FileNode] = []
        self._semantic_nodes: list[SemanticNode] = []

    @classmethod
    def create(cls, index_path: str | Path, config: Config | None = None) -> "DocSearchIndex":
        """Create a new index at the given path."""
        if config is None:
            config = Config()
        config.index = IndexConfig(path=Path(index_path))
        return cls(config)

    @classmethod
    def open(cls, index_path: str | Path, config: Config | None = None) -> "DocSearchIndex":
        """Open an existing index."""
        index_path = Path(index_path)
        if config is None:
            config = Config()
            config.index = IndexConfig(path=index_path)

        # Load metadata to verify
        metadata = load_metadata(index_path)
        if metadata is None:
            raise ValueError(f"No index found at {index_path}")

        # Verify model compatibility
        from docsearch.model import EmbeddingModel
        model = EmbeddingModel(config.model)
        if model.embedding_dim != metadata.embedding_dim:
            raise ValueError(f"Model dimension mismatch: {model.embedding_dim} != {metadata.embedding_dim}")

        # Load embeddings and nodes
        from docsearch.storage import (
            get_connection,
            load_embeddings,
            load_file_nodes,
            load_semantic_nodes,
        )
        embeddings = load_embeddings(index_path)
        if embeddings is None:
            raise ValueError(f"Embeddings file not found at {index_path}")

        with get_connection(index_path) as conn:
            file_nodes = load_file_nodes(conn)
            semantic_nodes = load_semantic_nodes(conn)

        instance = cls(config)
        instance._embeddings = embeddings
        instance._file_nodes = file_nodes
        instance._semantic_nodes = semantic_nodes
        return instance

    def index_directory(self, docs_path: str | Path, force_rebuild: bool = False) -> IndexMetadata:
        """Index a documentation directory."""
        from docsearch.index import build_index
        metadata = build_index(Path(docs_path), self.config.index.path, self.config, force_rebuild)
        # Reload embeddings and nodes after indexing
        from docsearch.storage import get_connection, load_file_nodes
        self._embeddings = load_embeddings(self.config.index.path)
        with get_connection(self.config.index.path) as conn:
            self._file_nodes = load_file_nodes(conn)
            self._semantic_nodes = load_semantic_nodes(conn)
        return metadata

    def save(self):
        """Save the index (embeddings are saved automatically during indexing)."""
        # Embeddings are already saved by build_index
        pass

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the index."""
        if self._embeddings is None:
            raise ValueError("Index not loaded. Call open() or index_directory() first.")
        return search(query, self.config.index.path, self.config, top_k=top_k)


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file content."""
    content = filepath.read_bytes()
    return hashlib.sha256(content).hexdigest()


def index_file(
    filepath: Path,
    root: Path,
    config: Config,
    model: EmbeddingModel,
    conn,
    embeddings_list: list[np.ndarray],
) -> tuple[FileNode, list[SemanticNode]]:
    """Index a single Markdown file."""
    rel_path = str(filepath.relative_to(root))
    content = filepath.read_text(encoding="utf-8")
    content_hash = hash_content(content)

    # Parse and chunk
    parsed = parse_markdown(rel_path, content)
    file_node, nodes = chunk_document(parsed, rel_path, config.chunking)

    # Update file node hash
    file_node.content_hash = content_hash

    # Generate embeddings for nodes
    if nodes:
        retrieval_texts = [n.retrieval_text for n in nodes]
        node_embeddings = model.encode_documents(retrieval_texts, show_progress=False)

        for node, emb in zip(nodes, node_embeddings, strict=True):
            idx = len(embeddings_list)
            embeddings_list.append(emb)
            node.embedding = emb
            save_semantic_node(conn, node, idx)

        # Create file-level embedding as mean of children
        file_emb = np.mean(node_embeddings, axis=0)
        file_emb = file_emb / np.linalg.norm(file_emb)
        file_idx = len(embeddings_list)
        embeddings_list.append(file_emb)
        file_node.embedding = file_emb
    else:
        file_idx = None

    save_file_node(conn, file_node, file_idx)
    return file_node, nodes


def build_index(
    docs_path: Path,
    index_path: Path,
    config: Config,
    force_rebuild: bool = False,
) -> IndexMetadata:
    """Build or update the search index."""
    docs_path = Path(docs_path).resolve()
    index_path = Path(index_path).resolve()

    if force_rebuild and index_path.exists():
        import shutil
        shutil.rmtree(index_path)

    init_database(index_path, config)

    model = EmbeddingModel(config.model)
    embeddings_list = []

    start_time = time.time()

    with get_connection(index_path) as conn:
        # Get existing files
        existing_files = set(get_all_files(conn))

        # Find all Markdown files
        md_files = []
        for ext in config.markdown_extensions:
            md_files.extend(docs_path.rglob(f"*{ext}"))

        # Filter ignored directories
        filtered_files = []
        for f in md_files:
            if any(part in config.ignore_dirs for part in f.parts):
                continue
            filtered_files.append(f)

        md_files = filtered_files

        # Process files
        indexed_files = set()
        for filepath in md_files:
            rel_path = str(filepath.relative_to(docs_path))
            indexed_files.add(rel_path)

            # Check if file changed
            file_info = get_file_info(conn, rel_path)
            current_hash = compute_file_hash(filepath)

            if file_info and not force_rebuild and file_info["content_hash"] == current_hash:
                # Unchanged, load existing embeddings
                file_nodes = load_file_nodes(conn)
                file_node = next((fn for fn in file_nodes if fn.file == rel_path), None)
                if file_node and file_node.embedding_idx is not None:
                    # We'll load embeddings later
                    pass
                continue

            # Index or re-index file
            print(f"Indexing: {rel_path}")  # noqa: T201
            index_file(filepath, docs_path, config, model, conn, embeddings_list)

        # Remove deleted files
        deleted = existing_files - indexed_files
        for file in deleted:
            print(f"Removing: {file}")  # noqa: T201
            delete_file_nodes(conn, file)

        # Load existing embeddings for unchanged files
        if embeddings_list:
            all_embeddings = np.vstack(embeddings_list)
        else:
            all_embeddings = np.empty((0, model.embedding_dim), dtype=np.float32)

        # Save embeddings
        save_embeddings(index_path, all_embeddings)

        # Create metadata
        metadata = IndexMetadata(
            version=SCHEMA_VERSION,
            model_name=config.model.name,
            embedding_dim=model.embedding_dim,
            file_count=get_file_count(conn),
            node_count=get_node_count(conn),
            created_at=start_time,
            updated_at=time.time(),
            config_hash=hash_content(str(config)),
        )
        save_metadata(index_path, metadata)

    return metadata