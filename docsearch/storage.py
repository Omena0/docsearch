"""Storage layer for index persistence using SQLite and NumPy."""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from docsearch.config import Config
from docsearch.model import EmbeddingModel
from docsearch.types import FileNode, IndexMetadata, SemanticNode

SCHEMA_VERSION = 1
EMBEDDING_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"


def get_db_path(index_path: Path) -> Path:
    return index_path / "index.sqlite"


def get_embedding_path(index_path: Path) -> Path:
    return index_path / EMBEDDING_FILE


def get_metadata_path(index_path: Path) -> Path:
    return index_path / METADATA_FILE


@contextmanager
def get_connection(index_path: Path):
    """Get database connection with row factory."""
    db_path = get_db_path(index_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(index_path: Path, config: Config):
    """Initialize database schema."""
    index_path.mkdir(parents=True, exist_ok=True)
    db_path = get_db_path(index_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                file TEXT NOT NULL UNIQUE,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                child_count INTEGER DEFAULT 0,
                embedding_idx INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                file TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                heading_level INTEGER NOT NULL,
                heading_text TEXT,
                heading_path TEXT NOT NULL,
                parent_id TEXT,
                content_hash TEXT NOT NULL,
                retrieval_text TEXT NOT NULL,
                source_text TEXT NOT NULL,
                level INTEGER DEFAULT 0,
                embedding_idx INTEGER,
                weight REAL DEFAULT 1.0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level)")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                file, heading_text, heading_path, retrieval_text, source_text,
                content='nodes', content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, file, heading_text, heading_path,
                    retrieval_text, source_text)
                VALUES (new.rowid, new.file, new.heading_text, new.heading_path,
                    new.retrieval_text, new.source_text);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, file, heading_text, heading_path,
                    retrieval_text, source_text)
                VALUES ('delete', old.rowid, old.file, old.heading_text, old.heading_path,
                    old.retrieval_text, old.source_text);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, file, heading_text, heading_path,
                    retrieval_text, source_text)
                VALUES ('delete', old.rowid, old.file, old.heading_text, old.heading_path,
                    old.retrieval_text, old.source_text);
                INSERT INTO nodes_fts(rowid, file, heading_text, heading_path,
                    retrieval_text, source_text)
                VALUES (new.rowid, new.file, new.heading_text, new.heading_path,
                    new.retrieval_text, new.source_text);
            END
        """)


def save_metadata(index_path: Path, metadata: IndexMetadata):
    """Save index metadata."""
    meta_path = get_metadata_path(index_path)
    with open(meta_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def load_metadata(index_path: Path) -> IndexMetadata | None:
    """Load index metadata."""
    meta_path = get_metadata_path(index_path)
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        data = json.load(f)
    return IndexMetadata.from_dict(data)


def save_embeddings(index_path: Path, embeddings: np.ndarray):
    """Save embeddings to numpy file atomically."""
    emb_path = get_embedding_path(index_path)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = emb_path.with_suffix(".tmp")
    np.save(tmp_path, embeddings)
    os.replace(str(tmp_path)+'.npy', emb_path)


def load_embeddings(index_path: Path) -> np.ndarray | None:
    """Load embeddings from numpy file."""
    emb_path = get_embedding_path(index_path)
    if not emb_path.exists():
        return None
    return np.load(emb_path)


def open_index(
    index_path: Path, config: Config | None = None
) -> tuple[Config, np.ndarray, list[FileNode], list[SemanticNode]]:
    """Open an existing index."""
    index_path = Path(index_path).resolve()

    metadata = load_metadata(index_path)
    if metadata is None:
        raise ValueError(f"No index found at {index_path}")

    if config is None:
        config = Config()
        # Use a simple object with path and embedding_dim for compatibility
        config.index = type(
            'obj', (object,),
            {'path': index_path, 'embedding_dim': metadata.embedding_dim}
        )()

    # Verify model compatibility
    model = EmbeddingModel(config.model)
    if model.embedding_dim != metadata.embedding_dim:
        raise ValueError(
            f"Model dimension mismatch: {model.embedding_dim} != {metadata.embedding_dim}"
        )

    # Load embeddings
    embeddings = load_embeddings(index_path)
    if embeddings is None:
        raise ValueError(f"Embeddings file not found at {index_path}")

    with get_connection(index_path) as conn:
        file_nodes = load_file_nodes(conn)
        semantic_nodes = load_semantic_nodes(conn)

    return config, embeddings, file_nodes, semantic_nodes


def save_file_node(
    conn: sqlite3.Connection, node: FileNode, embedding_idx: int | None = None
):
    """Save file node to database."""
    conn.execute("""
        INSERT OR REPLACE INTO files
            (id, file, start_line, end_line, content_hash, mtime,
             size, child_count, embedding_idx)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        node.id, node.file, node.start_line, node.end_line,
        node.content_hash, 0, 0, node.child_count, embedding_idx,
    ))


def load_file_nodes(conn: sqlite3.Connection) -> list[FileNode]:
    """Load all file nodes."""
    rows = conn.execute("SELECT * FROM files").fetchall()
    nodes = []
    for row in rows:
        node = FileNode(
            id=row["id"],
            file=row["file"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            heading_path=(),
            content_hash=row["content_hash"],
            child_count=row["child_count"],
            embedding_idx=row["embedding_idx"],
        )
        nodes.append(node)
    return nodes


def save_semantic_node(
    conn: sqlite3.Connection, node: SemanticNode, embedding_idx: int | None = None
):
    """Save semantic node to database."""
    conn.execute("""
        INSERT OR REPLACE INTO nodes
            (id, file, start_line, end_line, heading_level, heading_text,
             heading_path, parent_id, content_hash, retrieval_text,
             source_text, level, embedding_idx, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        node.id,
        node.file,
        node.start_line,
        node.end_line,
        node.heading_level,
        node.heading_text,
        " > ".join(node.heading_path),
        node.parent_id,
        node.content_hash,
        node.retrieval_text,
        node.source_text,
        node.level,
        embedding_idx,
        node.weight,
    ))


def load_semantic_nodes(conn: sqlite3.Connection, file: str | None = None) -> list[SemanticNode]:
    """Load semantic nodes, optionally filtered by file."""
    if file:
        rows = conn.execute("SELECT * FROM nodes WHERE file = ?", (file,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM nodes").fetchall()

    nodes = []
    for row in rows:
        node = SemanticNode(
            id=row["id"],
            file=row["file"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            heading_level=row["heading_level"],
            heading_text=row["heading_text"],
            heading_path=tuple(row["heading_path"].split(" > ")) if row["heading_path"] else (),
            parent_id=row["parent_id"],
            content_hash=row["content_hash"],
            retrieval_text=row["retrieval_text"],
            source_text=row["source_text"],
            level=row["level"],
            embedding_idx=row["embedding_idx"],
            weight=row["weight"] if row["weight"] is not None else 1.0,
        )
        nodes.append(node)
    return nodes


def delete_file_nodes(conn: sqlite3.Connection, file: str):
    """Delete all nodes for a file."""
    conn.execute("DELETE FROM nodes WHERE file = ?", (file,))
    conn.execute("DELETE FROM files WHERE file = ?", (file,))


def get_file_info(conn: sqlite3.Connection, file: str) -> dict | None:
    """Get file info from database."""
    row = conn.execute(
        "SELECT content_hash, mtime, size FROM files WHERE file = ?", (file,)
    ).fetchone()
    if row:
        return {"content_hash": row["content_hash"], "mtime": row["mtime"], "size": row["size"]}
    return None


def get_all_files(conn: sqlite3.Connection) -> list[str]:
    """Get all indexed files."""
    rows = conn.execute("SELECT file FROM files").fetchall()
    return [row["file"] for row in rows]


def get_node_count(conn: sqlite3.Connection) -> int:
    """Get total node count."""
    row = conn.execute("SELECT COUNT(*) as cnt FROM nodes").fetchone()
    return row["cnt"] if row else 0


def get_file_count(conn: sqlite3.Connection) -> int:
    """Get total file count."""
    row = conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()
    return row["cnt"] if row else 0


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 100) -> list[tuple[str, float]]:
    """Full-text search using FTS5."""
    # Escape FTS5 special characters
    escaped = query.replace('"', '""')
    # Use phrase search for better identifier matching
    fts_query = f'"{escaped}"'

    rows = conn.execute("""
        SELECT n.file, n.id, n.start_line, n.end_line, n.heading_level,
               n.heading_text, n.heading_path,
               n.parent_id, n.content_hash, n.retrieval_text,
               n.source_text, n.level,
               bm25(nodes_fts) as rank
        FROM nodes_fts
        JOIN nodes n ON nodes_fts.rowid = n.rowid
        WHERE nodes_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (fts_query, limit)).fetchall()

    results = []
    for row in rows:
        # Convert bm25 to similarity (lower bm25 = better match)
        score = 1.0 / (1.0 + row["rank"])
        results.append((row["id"], score))
    return results