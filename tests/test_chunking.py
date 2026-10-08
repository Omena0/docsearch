"""Tests for chunking."""

from docsearch.chunking import (
    build_retrieval_text,
    chunk_document,
    estimate_tokens,
    split_large_section,
)
from docsearch.config import ChunkingConfig
from docsearch.markdown import parse_markdown
from docsearch.types import SemanticNode


def test_chunk_normal_section():
    content = """# H1
Some content here.
## H2
More content.
"""
    parsed = parse_markdown("test.md", content)
    config = ChunkingConfig(target_tokens=400, overlap_tokens=50)
    file_node, nodes = chunk_document(parsed, "test.md", config)

    assert file_node.file == "test.md"
    assert file_node.child_count == 2
    assert len(nodes) == 2
    assert nodes[0].heading_text == "H1"
    assert nodes[1].heading_text == "H2"


def test_chunk_preserves_heading_path():
    content = """# Kernel
## Processes
### fork
"""
    parsed = parse_markdown("test.md", content)
    config = ChunkingConfig(target_tokens=400, overlap_tokens=50)
    file_node, nodes = chunk_document(parsed, "test.md", config)

    fork_node = next(n for n in nodes if n.heading_text == "fork")
    assert fork_node.heading_path == ("Kernel", "Processes", "fork")
    assert fork_node.level == 3


def test_chunk_preserves_source_range():
    content = """# H1
Line 1
Line 2
## H2
Line 4
"""
    parsed = parse_markdown("test.md", content)
    config = ChunkingConfig(target_tokens=400, overlap_tokens=50)
    file_node, nodes = chunk_document(parsed, "test.md", config)

    h1_node = next(n for n in nodes if n.heading_text == "H1")
    assert h1_node.start_line == 2
    assert h1_node.end_line == 3  # Before H2

    h2_node = next(n for n in nodes if n.heading_text == "H2")
    assert h2_node.start_line == 5
    assert h2_node.end_line == 5


def test_split_large_section():
    # Create a large node - need enough content to exceed max_tokens
    # Use multiple lines so splitting works
    large_text = "\n".join(["x " * 100] * 50)  # 50 lines, ~5000 chars total
    node = SemanticNode(
        id="test:1",
        file="test.md",
        start_line=1,
        end_line=50,
        heading_level=2,
        heading_text="Large",
        heading_path=("Root", "Large"),
        parent_id="file:test.md",
        retrieval_text=("Root > Large\n\n" + large_text),
        source_text=large_text,
        content_hash="abc",
        level=2,
    )
    config = ChunkingConfig(target_tokens=100, overlap_tokens=20, max_tokens=512)
    chunks = split_large_section(node, config)

    assert len(chunks) > 1
    # Check overlap
    for i in range(len(chunks) - 1):
        # Next chunk should start before previous ends
        assert chunks[i+1].start_line <= chunks[i].end_line


def test_retrieval_text_includes_heading_path():
    text = build_retrieval_text(("A", "B", "C"), "content here")
    assert "A > B > C" in text
    assert "content here" in text


def test_estimate_tokens():
    # Roughly 4 chars per token
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 0


def test_chunk_overlap_behavior():
    content = "# H1\n" + "\n".join(["x " * 100] * 50)  # Large section with multiple lines
    parsed = parse_markdown("test.md", content)
    config = ChunkingConfig(target_tokens=100, overlap_tokens=20, max_tokens=512)
    file_node, nodes = chunk_document(parsed, "test.md", config)

    # Should have multiple chunks for the large section
    h1_chunks = [n for n in nodes if n.heading_text == "H1"]
    assert len(h1_chunks) >= 2

    # Check that chunks have overlapping line ranges
    for i in range(len(h1_chunks) - 1):
        assert h1_chunks[i+1].start_line <= h1_chunks[i].end_line


def test_file_node_metadata():
    content = "# H1\ncontent"
    parsed = parse_markdown("test.md", content)
    config = ChunkingConfig(target_tokens=400, overlap_tokens=50)
    file_node, nodes = chunk_document(parsed, "test.md", config)

    assert file_node.id == "file:test.md"
    assert file_node.start_line == 1
    assert file_node.end_line == 2
    assert file_node.child_count == 1