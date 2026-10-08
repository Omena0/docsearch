"""Tests for storage layer."""

import tempfile
from pathlib import Path

import numpy as np

from docsearch.config import Config, IndexConfig
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
    search_fts,
)
from docsearch.types import FileNode, IndexMetadata, SemanticNode


def test_init_database():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        assert (index_path / "index.sqlite").exists()


def test_save_load_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        metadata = IndexMetadata(
            version=1,
            model_name="test-model",
            embedding_dim=384,
            file_count=10,
            node_count=100,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            config_hash="abc123",
        )
        save_metadata(index_path, metadata)
        loaded = load_metadata(index_path)

        assert loaded is not None
        assert loaded.version == 1
        assert loaded.model_name == "test-model"
        assert loaded.embedding_dim == 384
        assert loaded.file_count == 10
        assert loaded.node_count == 100


def test_save_load_embeddings():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        embeddings = np.random.rand(10, 384).astype(np.float32)
        save_embeddings(index_path, embeddings)
        loaded = load_embeddings(index_path)

        assert loaded is not None
        assert loaded.shape == (10, 384)
        np.testing.assert_array_almost_equal(loaded, embeddings)


def test_file_node_crud():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        with get_connection(index_path) as conn:
            node = FileNode(
                id="file:test.md",
                file="test.md",
                start_line=1,
                end_line=100,
                heading_path=(),
                content_hash="abc123",
                child_count=5,
            )
            save_file_node(conn, node, embedding_idx=0)

            loaded = load_file_nodes(conn)
            assert len(loaded) == 1
            assert loaded[0].file == "test.md"
            assert loaded[0].child_count == 5


def test_semantic_node_crud():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        with get_connection(index_path) as conn:
            node = SemanticNode(
                id="test.md:10",
                file="test.md",
                start_line=10,
                end_line=20,
                heading_level=2,
                heading_text="Test Heading",
                heading_path=("Root", "Test Heading"),
                parent_id="file:test.md",
                content_hash="def456",
                retrieval_text="Root > Test Heading\n\ncontent",
                source_text="content",
                level=2,
            )
            save_semantic_node(conn, node, embedding_idx=1)

            loaded = load_semantic_nodes(conn, file="test.md")
            assert len(loaded) == 1
            assert loaded[0].heading_text == "Test Heading"
            assert loaded[0].heading_path == ("Root", "Test Heading")
            assert loaded[0].parent_id == "file:test.md"


def test_delete_file_nodes():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        with get_connection(index_path) as conn:
            # Add file and nodes
            file_node = FileNode(id="file:test.md", file="test.md", start_line=1, end_line=10, heading_path=(), content_hash="abc", child_count=1)
            save_file_node(conn, file_node)
            node = SemanticNode(id="test.md:1", file="test.md", start_line=1, end_line=10, heading_level=1, heading_text="H1", heading_path=("H1",), parent_id="file:test.md", content_hash="abc", retrieval_text="H1\n\ncontent", source_text="content", level=1)
            save_semantic_node(conn, node)

            assert len(load_file_nodes(conn)) == 1
            assert len(load_semantic_nodes(conn)) == 1

            delete_file_nodes(conn, "test.md")

            assert len(load_file_nodes(conn)) == 0
            assert len(load_semantic_nodes(conn)) == 0


def test_get_file_info():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        with get_connection(index_path) as conn:
            node = FileNode(id="file:test.md", file="test.md", start_line=1, end_line=10, heading_path=(), content_hash="abc123", child_count=1)
            save_file_node(conn, node)

            info = get_file_info(conn, "test.md")
            assert info is not None
            assert info["content_hash"] == "abc123"

            info = get_file_info(conn, "nonexistent.md")
            assert info is None


def test_get_all_files():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        with get_connection(index_path) as conn:
            for f in ["a.md", "b.md", "c.md"]:
                node = FileNode(id=f"file:{f}", file=f, start_line=1, end_line=10, heading_path=(), content_hash="abc", child_count=0)
                save_file_node(conn, node)

            files = get_all_files(conn)
            assert set(files) == {"a.md", "b.md", "c.md"}


def test_counts():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        with get_connection(index_path) as conn:
            assert get_file_count(conn) == 0
            assert get_node_count(conn) == 0

            for i in range(3):
                f = f"file{i}.md"
                file_node = FileNode(id=f"file:{f}", file=f, start_line=1, end_line=10, heading_path=(), content_hash="abc", child_count=2)
                save_file_node(conn, file_node)
                for j in range(2):
                    node = SemanticNode(id=f"{f}:{j}", file=f, start_line=j+1, end_line=j+5, heading_level=1, heading_text=f"H{j}", heading_path=(f"H{j}",), parent_id=f"file:{f}", content_hash="abc", retrieval_text="text", source_text="text", level=1)
                    save_semantic_node(conn, node)

            assert get_file_count(conn) == 3
            assert get_node_count(conn) == 6


def test_fts_search():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        config = Config()
        config.index = IndexConfig(path=index_path)
        init_database(index_path, config)

        with get_connection(index_path) as conn:
            # Add nodes with searchable content
            node1 = SemanticNode(id="a.md:1", file="a.md", start_line=1, end_line=10, heading_level=1, heading_text="Process Creation", heading_path=("Process Creation",), parent_id="file:a.md", content_hash="abc", retrieval_text="Process Creation\n\nfork creates a child process", source_text="fork creates a child process", level=1)
            node2 = SemanticNode(id="b.md:1", file="b.md", start_line=1, end_line=10, heading_level=1, heading_text="Memory Management", heading_path=("Memory Management",), parent_id="file:b.md", content_hash="def", retrieval_text="Memory Management\n\npage cache stores file data", source_text="page cache stores file data", level=1)
            save_semantic_node(conn, node1)
            save_semantic_node(conn, node2)

            # Search for "fork"
            results = search_fts(conn, "fork", limit=10)
            assert len(results) == 1
            assert results[0][0] == "a.md:1"

            # Search for "page"
            results = search_fts(conn, "page", limit=10)
            assert len(results) == 1
            assert results[0][0] == "b.md:1"

            # Search for "process" (should match node1 which has "process" in retrieval_text)
            results = search_fts(conn, "process", limit=10)
            assert len(results) == 1
            assert results[0][0] == "a.md:1"