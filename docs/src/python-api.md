# Python API

docsearch provides a Python API for embedding search in your applications.

## Quick Start

```python
from docsearch import DocSearchIndex
from docsearch.config import Config

# Create or open an index
index = DocSearchIndex.create("./docs.index")

# Index documentation
metadata = index.index_directory("./my-docs")
print(f"Indexed {metadata.file_count} files, {metadata.node_count} nodes")

# Search
results = index.search("how does fork work?", top_k=5)
for r in results:
    print(f"{r.file}:{r.start_line}-{r.end_line}")
    print(f"  {r.heading_str}")
    print(f"  {r.markdown_context[:200]}...")
    print(f"  score={r.score:.4f}")
```

## DocSearchIndex

### Class Methods

#### `DocSearchIndex.create(index_path, config=None)`

Create a new index at the given path.

```python
index = DocSearchIndex.create("./docs.index", config)
```

**Parameters:**
- `index_path` (str|Path): Path to index directory
- `config` (Config, optional): Configuration object

**Returns:** `DocSearchIndex` instance

#### `DocSearchIndex.open(index_path, config=None)`

Open an existing index.

```python
index = DocSearchIndex.open("./docs.index")
```

**Parameters:**
- `index_path` (str|Path): Path to index directory
- `config` (Config, optional): Configuration object

**Returns:** `DocSearchIndex` instance

### Instance Methods

#### `index_directory(docs_path, force_rebuild=False)`

Index a documentation directory.

```python
metadata = index.index_directory("./docs", force_rebuild=True)
```

**Parameters:**
- `docs_path` (str|Path): Path to documentation directory
- `force_rebuild` (bool): Force full rebuild

**Returns:** `IndexMetadata`

#### `search(query, top_k=5)`

Search the index.

```python
results = index.search("query", top_k=10)
```

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results

**Returns:** `list[SearchResult]`

## SearchResult

Each search result contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `file` | str | Source file path |
| `start_line` | int | Start line number |
| `end_line` | int | End line number |
| `score` | float | Similarity score (0-1) |
| `heading_path` | tuple[str] | Heading hierarchy |
| `matched_node_id` | str | Internal node ID |
| `matched_level` | int | Heading level (1-6) |
| `matched_text` | str | Truncated retrieval text |
| `source_text` | str | Truncated source text |
| `markdown_context` | str | Full hierarchical markdown context |
| `heading_str` | property | `"> ".join(heading_path)` |
| `location` | property | `"file:start-end"` |

## Configuration

```python
from docsearch.config import (
    Config, ModelConfig, ChunkingConfig,
    IndexConfig, SearchConfig
)

config = Config(
    model=ModelConfig(
        name="BAAI/bge-small-en-v1.5",
        device="auto",
        batch_size=128,
    ),
    chunking=ChunkingConfig(
        target_tokens=400,
        overlap_tokens=50,
        max_tokens=512,
    ),
    index=IndexConfig(path="./docs.index"),
    search=SearchConfig(
        top_k=5,
        top_files=10,
        top_nodes=50,
        lexical_boost=0.1,
        min_score=0.0,
        context_expansion="parent_heading",
    ),
    ignore_dirs=(".git", "node_modules", "__pycache__"),
    markdown_extensions=(".md", ".markdown"),
    source_extensions=(".py", ".js", ".rs", ".go", ...),
    source_weight=0.3,
    auto_reindex=True,
)
```

## Low-Level Search

```python
from docsearch.search import search
from docsearch.storage import open_index

# Manual search with full config
config = Config()
config.index = IndexConfig(path="./docs.index")

results = search("query", "./docs.index", config, top_k=10)
```

## Using with Custom Models

```python
from docsearch.config import ModelConfig

config = Config(
    model=ModelConfig(
        name="BAAI/bge-base-en-v1.5",  # Different model
        device="cuda",
    ),
)
# Must use matching embedding dimension in IndexConfig
config.index = IndexConfig(path="./docs.index", embedding_dim=768)
```

## Async / Threading

docsearch is thread-safe for reads. For concurrent writes, use a single indexer.

```python
# Thread-safe reads
import threading

def search_worker(query, results):
    results.append(index.search(query))

threads = [threading.Thread(target=search_worker, args=(q, results)) for q in queries]
for t in threads: t.start()
for t in threads: t.join()
```

## Error Handling

```python
from docsearch import DocSearchIndex
from docsearch.exceptions import DocSearchError

try:
    index = DocSearchIndex.open("./docs.index")
    results = index.search("query")
except FileNotFoundError:
    print("Index not found - run index_directory first")
except DocSearchError as e:
    print(f"Search error: {e}")
```

## Type Hints

All public APIs are fully typed:

```python
from docsearch.types import SearchResult, SemanticNode, FileNode, IndexMetadata
```