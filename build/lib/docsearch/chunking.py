"""Chunking of Markdown sections into semantic nodes."""

import hashlib

from docsearch.config import ChunkingConfig
from docsearch.markdown import ParsedDocument, extract_section_text
from docsearch.types import FileNode, SemanticNode

# Approximate token estimation: 1 token ≈ 4 characters for English
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count."""
    return len(text) // CHARS_PER_TOKEN


def hash_content(text: str) -> str:
    """Compute content hash."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_retrieval_text(heading_path: tuple[str, ...], content: str) -> str:
    """Build retrieval text with heading context."""
    return " > ".join(heading_path) + "\n\n" + content if heading_path else content


def create_file_node(
    file: str,
    lines: list[str],
    content_hash: str,
    embedding: list[float] | None = None,
) -> FileNode:
    """Create a file-level node."""
    return FileNode(
        id=f"file:{file}",
        file=file,
        start_line=1,
        end_line=len(lines),
        heading_path=(),
        content_hash=content_hash,
        embedding=embedding,
        child_count=0,
    )


def create_heading_nodes(
    parsed: ParsedDocument,
    file: str,
    config: ChunkingConfig,
) -> list[SemanticNode]:
    """Create semantic nodes from parsed headings."""
    nodes = []

    for heading in parsed.headings:
        # Section content starts after the heading line
        start_line = heading.start_line + 1
        end_line = heading.end_line
        if start_line > end_line:
            # Empty section
            source_text = ""
        else:
            source_text = extract_section_text(parsed, start_line, end_line)
        content_hash = hash_content(source_text)

        retrieval_text = build_retrieval_text(heading.path, source_text)

        node = SemanticNode(
            id=f"{file}:{heading.start_line}",
            file=file,
            start_line=start_line,
            end_line=end_line,
            heading_level=heading.level,
            heading_text=heading.text,
            heading_path=heading.path,
            parent_id=f"{file}:{heading.parent.start_line}" if heading.parent else f"file:{file}",
            retrieval_text=retrieval_text,
            source_text=source_text,
            content_hash=content_hash,
            level=heading.level,
        )
        nodes.append(node)

    return nodes


def split_large_section(
    node: SemanticNode,
    config: ChunkingConfig,
) -> list[SemanticNode]:
    """Split a large section into overlapping chunks."""
    if estimate_tokens(node.retrieval_text) <= config.max_tokens:
        return [node]

    chunks = []
    lines = node.source_text.splitlines()
    target_chars = config.target_tokens * CHARS_PER_TOKEN

    if len(lines) <= 1:
        return [node]

    start_idx = 0
    chunk_num = 0

    while start_idx < len(lines):
        # Find end index
        end_idx = start_idx
        char_count = 0

        while end_idx < len(lines) and char_count < target_chars:
            char_count += len(lines[end_idx]) + 1  # +1 for newline
            end_idx += 1

        # Extract chunk
        chunk_lines = lines[start_idx:end_idx]
        chunk_source = "\n".join(chunk_lines)

        # Calculate actual line numbers
        chunk_start = node.start_line + start_idx
        chunk_end = node.start_line + end_idx - 1

        chunk_retrieval = build_retrieval_text(node.heading_path, chunk_source)

        chunk = SemanticNode(
            id=f"{node.id}:chunk{chunk_num}",
            file=node.file,
            start_line=chunk_start,
            end_line=chunk_end,
            heading_level=node.heading_level,
            heading_text=node.heading_text,
            heading_path=node.heading_path,
            parent_id=node.parent_id,
            retrieval_text=chunk_retrieval,
            source_text=chunk_source,
            content_hash=hash_content(chunk_source),
            level=node.level,
        )
        chunks.append(chunk)
        chunk_num += 1

        # Move start with overlap
        chunk_size = max(1, end_idx - start_idx)
        overlap_lines = max(0, chunk_size * config.overlap_tokens // config.target_tokens)
        new_start = end_idx - overlap_lines
        # Ensure we always make progress
        if new_start >= start_idx:
            new_start = start_idx + 1
        start_idx = new_start
        if start_idx >= len(lines):
            break

    return chunks


def chunk_document(
    parsed: ParsedDocument,
    file: str,
    config: ChunkingConfig,
) -> tuple[FileNode, list[SemanticNode]]:
    """Chunk a parsed document into file node and semantic nodes."""
    content = "\n".join(parsed.lines)
    content_hash = hash_content(content)
    file_node = create_file_node(file, parsed.lines, content_hash)

    heading_nodes = create_heading_nodes(parsed, file, config)

    # Split large sections
    all_nodes = []
    for node in heading_nodes:
        chunks = split_large_section(node, config)
        all_nodes.extend(chunks)

    file_node.child_count = len(all_nodes)
    return file_node, all_nodes