"""Tests for search functionality."""

import tempfile
from pathlib import Path

import numpy as np

from docsearch.config import ChunkingConfig, Config, IndexConfig, ModelConfig, SearchConfig
from docsearch.index import build_index
from docsearch.search import (
    cosine_similarity,
    expand_context,
    merge_overlapping_results,
    search,
    search_files,
    search_nodes,
)
from docsearch.types import FileNode, SearchResult, SemanticNode


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    scores = cosine_similarity(a, b)
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_search_files():
    file_nodes = [
        FileNode(id="file:a.md", file="a.md", start_line=1, end_line=10, heading_path=(), content_hash="abc", child_count=1, embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32), embedding_idx=0),
        FileNode(id="file:b.md", file="b.md", start_line=1, end_line=10, heading_path=(), content_hash="def", child_count=1, embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32), embedding_idx=1),
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # a.md
        [0.0, 1.0, 0.0],  # b.md
    ], dtype=np.float32)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = search_files(query, file_nodes, embeddings, top_k=2)

    assert len(results) == 2
    assert results[0][0].file == "a.md"
    assert results[0][1] > results[1][1]


def test_search_nodes():
    nodes = [
        SemanticNode(id="a.md:1", file="a.md", start_line=1, end_line=10, heading_level=1, heading_text="H1", heading_path=("H1",), parent_id="file:a.md", content_hash="abc", retrieval_text="text", source_text="text", level=1, embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32), embedding_idx=0),
        SemanticNode(id="a.md:2", file="a.md", start_line=11, end_line=20, heading_level=2, heading_text="H2", heading_path=("H1", "H2"), parent_id="a.md:1", content_hash="def", retrieval_text="text", source_text="text", level=2, embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32), embedding_idx=1),
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = search_nodes(query, nodes, embeddings, top_k=2)

    assert len(results) == 2
    assert results[0][0].id == "a.md:1"
    assert results[0][1] > results[1][1]


def test_merge_overlapping_results():
    nodes = [
        SemanticNode(id="file:a.md", file="a.md", start_line=1, end_line=100, heading_level=0, heading_text="", heading_path=(), parent_id=None, content_hash="", retrieval_text="", source_text="", level=0),
        SemanticNode(id="a.md:1", file="a.md", start_line=1, end_line=80, heading_level=1, heading_text="H1", heading_path=("H1",), parent_id="file:a.md", content_hash="", retrieval_text="", source_text="", level=1),
        SemanticNode(id="a.md:2", file="a.md", start_line=10, end_line=50, heading_level=2, heading_text="H2", heading_path=("H1", "H2"), parent_id="a.md:1", content_hash="", retrieval_text="", source_text="", level=2),
        SemanticNode(id="a.md:3", file="a.md", start_line=30, end_line=40, heading_level=3, heading_text="H3", heading_path=("H1", "H2", "H3"), parent_id="a.md:2", content_hash="", retrieval_text="", source_text="", level=3),
    ]
    results = [
        SearchResult(file="a.md", start_line=10, end_line=50, score=0.9, heading_path=("H2",), matched_node_id="a.md:2", matched_level=2, node=nodes[2]),
        SearchResult(file="a.md", start_line=30, end_line=40, score=0.95, heading_path=("H2", "H3"), matched_node_id="a.md:3", matched_level=3, node=nodes[3]),
        SearchResult(file="b.md", start_line=10, end_line=20, score=0.8, heading_path=("H1",), matched_node_id="b.md:1", matched_level=1, node=nodes[0]),
    ]
    merged = merge_overlapping_results(results, nodes)

    # First two overlap (H2 is ancestor of H3), should keep higher scored (second one - H3)
    assert len(merged) == 2
    assert merged[0].matched_node_id == "a.md:3"
    assert merged[1].matched_node_id == "b.md:1"


def test_expand_context_parent_heading():
    nodes = [
        SemanticNode(id="file:a.md", file="a.md", start_line=1, end_line=100, heading_level=0, heading_text="", heading_path=(), parent_id=None, content_hash="", retrieval_text="", source_text="", level=0),
        SemanticNode(id="a.md:1", file="a.md", start_line=1, end_line=50, heading_level=1, heading_text="H1", heading_path=("H1",), parent_id="file:a.md", content_hash="", retrieval_text="", source_text="", level=1),
        SemanticNode(id="a.md:2", file="a.md", start_line=10, end_line=30, heading_level=2, heading_text="H2", heading_path=("H1", "H2"), parent_id="a.md:1", content_hash="", retrieval_text="", source_text="", level=2),
    ]
    file_nodes = [nodes[0]]

    child = nodes[2]
    start, end = expand_context(child, nodes, file_nodes, "parent_heading")

    # Should expand to parent heading start
    assert start == 1
    assert end == 30


def test_expand_context_full_ancestry():
    nodes = [
        SemanticNode(id="file:a.md", file="a.md", start_line=1, end_line=100, heading_level=0, heading_text="", heading_path=(), parent_id=None, content_hash="", retrieval_text="", source_text="", level=0),
        SemanticNode(id="a.md:1", file="a.md", start_line=1, end_line=80, heading_level=1, heading_text="H1", heading_path=("H1",), parent_id="file:a.md", content_hash="", retrieval_text="", source_text="", level=1),
        SemanticNode(id="a.md:2", file="a.md", start_line=10, end_line=50, heading_level=2, heading_text="H2", heading_path=("H1", "H2"), parent_id="a.md:1", content_hash="", retrieval_text="", source_text="", level=2),
        SemanticNode(id="a.md:3", file="a.md", start_line=20, end_line=30, heading_level=3, heading_text="H3", heading_path=("H1", "H2", "H3"), parent_id="a.md:2", content_hash="", retrieval_text="", source_text="", level=3),
    ]
    file_nodes = [nodes[0]]

    child = nodes[3]
    start, end = expand_context(child, nodes, file_nodes, "full_ancestry")

    # Should expand to root
    assert start == 1
    assert end == 30


def test_search_result_properties():
    result = SearchResult(
        file="docs/kernel.md",
        start_line=100,
        end_line=150,
        score=0.95,
        heading_path=("Kernel", "Processes", "fork"),
        matched_node_id="kernel.md:42",
        matched_level=3,
    )
    assert result.location == "docs/kernel.md:100-150"
    assert result.heading_str == "Kernel > Processes > fork"


def test_integration_search():
    """Integration test with real index."""
    with tempfile.TemporaryDirectory() as tmp:
        docs_path = Path(tmp) / "docs"
        docs_path.mkdir()
        index_path = Path(tmp) / "index"

        # Create test docs
        (docs_path / "kernel.md").write_text("""# Kernel
## Processes
### fork
The fork system call creates a child process.
### exec
The exec system call replaces the process image.
## Memory
### Page Cache
The page cache stores file data in memory.
""")
        (docs_path / "bootloader.md").write_text("""# Bootloader
## GRUB
GRUB loads the kernel.
""")

        config = Config()
        config.index = IndexConfig(path=index_path)
        config.model = ModelConfig(name="BAAI/bge-small-en-v1.5")
        config.chunking = ChunkingConfig(target_tokens=400, overlap_tokens=50)
        config.search = SearchConfig(top_k=5, top_files=10, top_nodes=20, lexical_boost=0.1)

        # Build index (will download model on first run)
        metadata = build_index(docs_path, index_path, config)
        assert metadata.file_count == 2
        assert metadata.node_count > 0

        # Search
        results = search("how does fork create a child process", index_path, config, top_k=3)

        assert len(results) > 0
        # Should find fork section
        found_fork = any("fork" in r.heading_str.lower() for r in results)
        assert found_fork, f"Expected fork in results: {[r.heading_str for r in results]}"

        # Check line numbers are valid
        for r in results:
            assert r.start_line >= 1
            assert r.end_line >= r.start_line
            assert r.score >= 0
            assert r.score <= 1.0


def test_incremental_indexing():
    """Test that re-indexing only updates changed files."""
    with tempfile.TemporaryDirectory() as tmp:
        docs_path = Path(tmp) / "docs"
        docs_path.mkdir()
        index_path = Path(tmp) / "index"

        (docs_path / "a.md").write_text("# A\nContent A")
        (docs_path / "b.md").write_text("# B\nContent B")

        config = Config()
        config.index = IndexConfig(path=index_path)
        config.model = ModelConfig(name="BAAI/bge-small-en-v1.5")

        # Initial index
        meta1 = build_index(docs_path, index_path, config)
        assert meta1.file_count == 2

        # Modify one file
        (docs_path / "a.md").write_text("# A\nModified Content A")

        # Re-index (should only update a.md)
        meta2 = build_index(docs_path, index_path, config)
        assert meta2.file_count == 2

        # Add new file
        (docs_path / "c.md").write_text("# C\nContent C")
        meta3 = build_index(docs_path, index_path, config)
        assert meta3.file_count == 3

        # Delete file
        (docs_path / "b.md").unlink()
        meta4 = build_index(docs_path, index_path, config)
        assert meta4.file_count == 2


def test_search_with_lexical_boost():
    with tempfile.TemporaryDirectory() as tmp:
        docs_path = Path(tmp) / "docs"
        docs_path.mkdir()
        index_path = Path(tmp) / "index"

        (docs_path / "kernel.md").write_text("""# Kernel
## task_struct
The task_struct is the process descriptor.
""")

        config = Config()
        config.index = IndexConfig(path=index_path)
        config.model = ModelConfig(name="BAAI/bge-small-en-v1.5")
        config.search = SearchConfig(top_k=5, lexical_boost=0.5)

        build_index(docs_path, index_path, config)

        # Search for exact identifier
        results = search("task_struct", index_path, config, top_k=3)

        assert len(results) > 0
        # Should rank the exact identifier match high
        found = any("task_struct" in r.heading_str for r in results)
        assert found