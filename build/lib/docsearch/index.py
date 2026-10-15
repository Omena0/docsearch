"""Main indexing logic."""

import hashlib
import time
from pathlib import Path

import numpy as np

from docsearch.cache import load_cached_embeddings, save_cached_embeddings
from docsearch.chunking import chunk_document, hash_content
from docsearch.config import Config, IndexConfig
from docsearch.markdown import parse_markdown
from docsearch.model import EmbeddingModel
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
            raise ValueError(
                f"Model dimension mismatch: {model.embedding_dim} != {metadata.embedding_dim}"
            )

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


def _filter_files(all_files: list[Path], docs_path: Path, config: Config) -> list[Path]:
    """Filter files by ignore_dirs, allowed_globs, and ignored_globs."""
    from docsearch.config_loader import should_index_file
    return [f for f in all_files if should_index_file(f, docs_path, config)]


def check_and_reindex(index_path: Path, config: Config):
    """Check if files have changed and reindex them if auto_reindex is enabled."""
    if not config.auto_reindex:
        return

    index_path = Path(index_path)
    metadata = load_metadata(index_path)
    if metadata is None:
        return  # No index to check

    docs_path_str = metadata.docs_path
    if docs_path_str is None:
        return  # No docs path stored

    docs_path = Path(docs_path_str)
    if not docs_path.exists():
        return  # Docs path doesn't exist

    with get_connection(index_path) as conn:
        # Get existing files
        existing_files = set(get_all_files(conn))

        # Find all files to index (Markdown + source code), respecting globs
        all_files = []
        for ext in config.markdown_extensions + config.source_extensions:
            all_files.extend(docs_path.rglob(f"*{ext}"))

        filtered_files = _filter_files(all_files, docs_path, config)

        # Dry run: check each file against cache dir
        files_to_update = []
        indexed_files = set()

        for filepath in filtered_files:
            rel_path = str(filepath.relative_to(docs_path))
            indexed_files.add(rel_path)

            # Check if file has cached embeddings
            current_hash = compute_file_hash(filepath)
            cached = load_cached_embeddings(index_path, current_hash)
            if cached is not None:
                # Already cached - skip, move on
                continue
            else:
                # Not cached - needs updating
                files_to_update.append((filepath, current_hash, rel_path))

        # Check for deleted files
        deleted = existing_files - indexed_files
        if deleted:
            files_to_update.append(None)  # signal deletions exist

        # If nothing to update and no deletions, return early
        if not files_to_update:
            return

        # Reindex changed/deleted files
        print("Auto-reindexing changed files...")  # noqa: T201
        build_index(docs_path, index_path, config, force_rebuild=False)

    def save(self):
        """Save the index (embeddings are saved automatically during indexing)."""
        # Embeddings are already saved by build_index
        pass

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the index."""
        if self._embeddings is None:
            raise ValueError("Index not loaded. Call open() or index_directory() first.")
        from docsearch.search import search as do_search
        return do_search(query, self.config.index.path, self.config, top_k=top_k)


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
    index_path: Path,
) -> tuple[FileNode, list[SemanticNode]]:
    """Index a single Markdown or source file."""
    rel_path = str(filepath.relative_to(root))
    is_source = filepath.suffix in config.source_extensions

    if is_source:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        content_hash = hash_content(content)
    else:
        content = filepath.read_text(encoding="utf-8")
        content_hash = hash_content(content)

    # Determine weight
    weight = config.source_weight if is_source else 1.0

    # Parse and chunk
    parsed = parse_markdown(rel_path, content)
    file_node, nodes = chunk_document(parsed, rel_path, config.chunking)

    # Update file node hash
    file_node.content_hash = content_hash
    file_node.weight = weight

    # Generate embeddings for nodes (with cache lookup)
    if nodes:
        # Look up cached embeddings for unchanged nodes
        cached_embeddings = {}
        texts_to_encode = []
        node_indices = []  # tracks which nodes need encoding
        for i, node in enumerate(nodes):
            cached = load_cached_embeddings(index_path, node.content_hash)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                texts_to_encode.append(node.retrieval_text)
                node_indices.append(i)

        # Compute embeddings for uncached nodes
        new_embeddings = []
        if texts_to_encode:
            new_embeddings = model.encode_documents(texts_to_encode, show_progress=False)

        # Build node embeddings list
        new_idx = 0
        for i, node in enumerate(nodes):
            if i in cached_embeddings:
                node.embedding = cached_embeddings[i]
            else:
                node.embedding = new_embeddings[new_idx]
                new_idx += 1
            node.weight = weight

        # Assemble embeddings list
        for node in nodes:
            embeddings_list.append(node.embedding)

        # Compute file-level embedding as mean of children
        file_emb = np.mean([n.embedding for n in nodes], axis=0)
        file_emb = file_emb / np.linalg.norm(file_emb)

        # Save file embedding
        embeddings_list.append(file_emb)

        # Save newly computed embeddings to cache
        for i, node in enumerate(nodes):
            if i not in cached_embeddings:
                save_cached_embeddings(index_path, node.content_hash, node.embedding)

        # Save semantic nodes to database
        for i, node in enumerate(nodes):
            save_semantic_node(conn, node, i)
    else:
        # Empty file - create zero vector for file embedding
        file_emb = np.zeros(model.embedding_dim, dtype=np.float32)
        embeddings_list.append(file_emb)

    # Save file node
    file_idx = len(embeddings_list) - 1
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

        # Find all files to index (Markdown + source code)
        all_files = []
        for ext in config.markdown_extensions + config.source_extensions:
            all_files.extend(docs_path.rglob(f"*{ext}"))

        
        filtered_files = _filter_files(all_files, docs_path, config)

        # Process files
        indexed_files = set()
        for filepath in filtered_files:
            rel_path = str(filepath.relative_to(docs_path))
            indexed_files.add(rel_path)

            # Check if file changed
            file_info = get_file_info(conn, rel_path)
            current_hash = compute_file_hash(filepath)

            if file_info and not force_rebuild and file_info["content_hash"] == current_hash:
                # Unchanged - load cached embeddings
                file_nodes = load_file_nodes(conn)
                semantic_nodes = load_semantic_nodes(conn)
                file_node = next((fn for fn in file_nodes if fn.file == rel_path), None)
                if file_node is not None:
                    # Load cached embeddings for this file
                    cached_file_emb = load_cached_embeddings(index_path, file_node.content_hash)
                    if cached_file_emb is not None:
                        embeddings_list.append(cached_file_emb)
                    else:
                        # Fallback: if file embedding not cached, load from DB
                        if file_node.embedding is not None:
                            embeddings_list.append(file_node.embedding)
                    # Load cached embeddings for semantic nodes
                    file_semantic_nodes = [n for n in semantic_nodes if n.file == rel_path]
                    for node in file_semantic_nodes:
                        cached = load_cached_embeddings(index_path, node.content_hash)
                        if cached is not None:
                            embeddings_list.append(cached)
                        elif node.embedding is not None:
                            embeddings_list.append(node.embedding)
                continue

            # Index or re-index file
            print(f"Indexing: {rel_path}")  # noqa: T201
            index_file(filepath, docs_path, config, model, conn, embeddings_list, index_path)

        # Remove deleted files
        deleted = existing_files - indexed_files
        for file in deleted:
            print(f"Removing: {file}")  # noqa: T201
            delete_file_nodes(conn, file)

        # Save embeddings
        if embeddings_list:
            all_embeddings = np.vstack(embeddings_list)
        else:
            all_embeddings = np.empty((0, model.embedding_dim), dtype=np.float32)

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
            docs_path=str(docs_path),
        )
        save_metadata(index_path, metadata)

    return metadata