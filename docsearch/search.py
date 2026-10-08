"""Search implementation with hierarchical retrieval."""

from pathlib import Path

import numpy as np

from docsearch.config import Config
from docsearch.model import EmbeddingModel
from docsearch.storage import get_connection, open_index, search_fts
from docsearch.types import FileNode, SearchResult, SemanticNode


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity (vectors must be normalized)."""
    return doc_vecs @ query_vec  # type: ignore[no-any-return]


def search_files(
    query_vec: np.ndarray,
    file_nodes: list[FileNode],
    embeddings: np.ndarray,
    top_k: int,
) -> list[tuple[FileNode, float]]:
    """Search file-level embeddings."""
    if not file_nodes:
        return []

    file_indices = [n.embedding_idx for n in file_nodes if n.embedding_idx is not None]
    valid_nodes = [n for n in file_nodes if n.embedding_idx is not None]

    if not file_indices:
        return []

    file_embeddings = embeddings[file_indices]
    scores = cosine_similarity(query_vec, file_embeddings)

    # Get top-k
    k = min(top_k, len(scores))
    if k == 0:
        return []
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    results = []
    for idx in top_indices:
        results.append((valid_nodes[idx], float(scores[idx])))

    return results


def search_nodes(
    query_vec: np.ndarray,
    nodes: list[SemanticNode],
    embeddings: np.ndarray,
    top_k: int,
    min_score: float = 0.0,
) -> list[tuple[SemanticNode, float]]:
    """Search semantic node embeddings."""
    if not nodes:
        return []

    node_indices = [n.embedding_idx for n in nodes if n.embedding_idx is not None]
    valid_nodes = [n for n in nodes if n.embedding_idx is not None]

    if not node_indices:
        return []

    node_embeddings = embeddings[node_indices]
    scores = cosine_similarity(query_vec, node_embeddings)

    # Filter by min_score
    mask = scores >= min_score
    if not np.any(mask):
        return []

    filtered_scores = scores[mask]
    filtered_nodes = [valid_nodes[i] for i, m in enumerate(mask) if m]

    k = min(top_k, len(filtered_scores))
    if k == 0:
        return []
    top_indices = np.argpartition(-filtered_scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-filtered_scores[top_indices])]

    results = []
    for idx in top_indices:
        results.append((filtered_nodes[idx], float(filtered_scores[idx])))

    return results


def merge_overlapping_results(
    results: list[SearchResult],
    semantic_nodes: list[SemanticNode],
) -> list[SearchResult]:
    """Deduplicate results by heading hierarchy, keeping the highest-scoring result per chain."""
    if not results:
        return []

    results = sorted(results, key=lambda r: r.score, reverse=True)

    node_lookup = {n.id: n for n in semantic_nodes}

    def is_ancestor(node_a: SemanticNode, node_b: SemanticNode) -> bool:
        """Check if node_a is an ancestor of node_b within the same file."""
        if node_a.file != node_b.file:
            return False
        parent_id = node_b.parent_id
        while parent_id:
            parent = node_lookup.get(parent_id)
            if parent is None:
                return False
            if parent.id == node_a.id:
                return True
            parent_id = parent.parent_id
        return False

    merged: list[SearchResult] = []
    for result in results:
        skip = False
        for existing in merged:
            if result.file != existing.file:
                continue
            if is_ancestor(result.node, existing.node) or is_ancestor(existing.node, result.node):
                skip = True
                break
        if not skip:
            merged.append(result)

    return merged


def expand_context(
    node: SemanticNode,
    all_nodes: list[SemanticNode],
    file_nodes: list[FileNode],
    policy: str = "parent_heading",
) -> tuple[int, int]:
    """Expand result to include context based on policy."""
    start = node.start_line
    end = node.end_line

    if policy == "none":
        return start, end

    # Find parent heading
    parent_id = node.parent_id
    while parent_id:
        parent = next((n for n in all_nodes if n.id == parent_id), None)
        if parent is None:
            # Check file node
            file_parent = next((f for f in file_nodes if f.id == parent_id), None)
            if file_parent:
                start = min(start, file_parent.start_line)
                break
        if parent:
            if policy == "parent_heading":
                start = min(start, parent.start_line)
                break
            elif policy == "full_ancestry":
                start = min(start, parent.start_line)
                parent_id = parent.parent_id
            else:
                break
        else:
            break

    return start, end


def get_toplevel_content(node: SemanticNode, all_nodes: list[SemanticNode]) -> str:
    """Get the toplevel content of a node (content before any sub-heading)."""
    children = [n for n in all_nodes if n.parent_id == node.id]
    if not children:
        return node.source_text
    
    earliest_child = min(children, key=lambda c: c.start_line)
    cutoff = earliest_child.start_line - node.start_line
    if cutoff <= 0:
        return ""
    
    lines = node.source_text.splitlines()
    return "\n".join(lines[:cutoff])


def build_markdown_context(matched_node: SemanticNode, all_nodes: list[SemanticNode]) -> str:
    """Build markdown context for a matched node with hierarchical heading content."""
    ancestors = []
    parent_id = matched_node.parent_id
    while parent_id:
        parent = next((n for n in all_nodes if n.id == parent_id), None)
        if parent is None:
            break
        ancestors.append(parent)
        parent_id = parent.parent_id
    ancestors.reverse()

    children = [n for n in all_nodes if n.parent_id == matched_node.id]

    parts = []

    for ancestor in ancestors:
        heading_line = "#" * ancestor.heading_level + " " + ancestor.heading_text
        toplevel_content = get_toplevel_content(ancestor, all_nodes)
        parts.append(heading_line)
        if toplevel_content.strip():
            parts.append(toplevel_content)

    heading_line = "#" * matched_node.heading_level + " " + matched_node.heading_text
    toplevel_content = get_toplevel_content(matched_node, all_nodes)
    parts.append(heading_line)
    if toplevel_content.strip():
        parts.append(toplevel_content)

    for child in children:
        heading_line = "#" * child.heading_level + " " + child.heading_text
        child_toplevel = get_toplevel_content(child, all_nodes)
        parts.append(heading_line)
        if child_toplevel.strip():
            parts.append(child_toplevel)

    return "\n\n".join(parts)


def hierarchical_search(
    query: str,
    config: Config,
    query_vec: np.ndarray,
    file_nodes: list[FileNode],
    semantic_nodes: list[SemanticNode],
    embeddings: np.ndarray,
) -> list[SearchResult]:
    """Perform hierarchical search."""
    search_config = config.search

    # Level 1: File-level search
    file_results = search_files(query_vec, file_nodes, embeddings, search_config.top_files)

    # Collect candidate nodes from top files
    candidate_nodes = []
    for file_node, file_score in file_results:
        file_semantic = [n for n in semantic_nodes if n.file == file_node.file]
        if file_semantic:
            # Search within file
            node_results = search_nodes(
                query_vec, file_semantic, embeddings,
                search_config.top_nodes, search_config.min_score
            )
            for node, score in node_results:
                # Combine file score and node score
                combined_score = score * 0.7 + file_score * 0.3
                candidate_nodes.append((node, combined_score))

    # Sort by combined score
    candidate_nodes.sort(key=lambda x: x[1], reverse=True)
    candidate_nodes = candidate_nodes[:search_config.top_nodes]

    # Level 2: Also search all nodes directly for completeness
    all_node_results = search_nodes(
        query_vec, semantic_nodes, embeddings,
        search_config.top_nodes * 2, search_config.min_score
    )

    # Combine and deduplicate
    seen_ids = set()
    combined: list[tuple[SemanticNode, float]] = []
    for node, score in candidate_nodes:
        if node.id not in seen_ids:
            combined.append((node, score))
            seen_ids.add(node.id)

    for node, score in all_node_results:
        if node.id not in seen_ids:
            combined.append((node, score))
            seen_ids.add(node.id)

    # Lexical boost (FTS)
    if search_config.lexical_boost > 0:
        assert config.index is not None
        with get_connection(config.index.path) as conn:
            fts_results = search_fts(conn, query, limit=search_config.top_nodes * 2)
            fts_dict = dict(fts_results)

            boosted = []
            for node, score in combined:
                lexical_score = fts_dict.get(node.id, 0.0)
                boosted_score = score + search_config.lexical_boost * lexical_score
                boosted.append((node, boosted_score))
            combined = boosted

    # Sort final
    combined.sort(key=lambda x: x[1], reverse=True)
    combined = combined[:search_config.top_k * 3]  # Keep more for merging

    # Create SearchResult objects
    results = []
    for node, score in combined:
        start, end = expand_context(node, semantic_nodes, file_nodes,
                                     search_config.context_expansion)
        markdown_context = build_markdown_context(node, semantic_nodes)

        result = SearchResult(
            file=node.file,
            start_line=start,
            end_line=end,
            score=score,
            heading_path=node.heading_path,
            matched_node_id=node.id,
            matched_level=node.level,
            matched_text=node.retrieval_text[:500],
            source_text=node.source_text[:500],
            markdown_context=markdown_context,
            node=node,
        )
        results.append(result)

    # Deduplicate overlapping results by heading hierarchy
    results = merge_overlapping_results(results, semantic_nodes)

    # Final top-k
    return results[:search_config.top_k]


def search(
    query: str,
    index_path: Path,
    config: Config | None = None,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Main search function."""
    if config is None:
        config = Config()
        config.index = type('obj', (object,), {'path': index_path})()

    if top_k is not None:
        config.search.top_k = top_k

    # Open index
    config, embeddings, file_nodes, semantic_nodes = open_index(index_path, config)

    # Encode query
    model = EmbeddingModel(config.model)
    query_vec = model.encode_queries([query])[0]

    # Search
    results = hierarchical_search(
        query, config, query_vec,
        file_nodes, semantic_nodes, embeddings
    )

    return results