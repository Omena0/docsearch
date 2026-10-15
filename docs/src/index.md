# docsearch

**Fast local semantic search for Markdown documentation and source code.**

docsearch is a semantic search engine designed for local documentation and codebases. It uses BGE embeddings to provide relevant search results with hierarchical context, making it easy to find exactly what you need in your docs or code.

## Features

- **Semantic search**: BGE-small-en-v1.5 embeddings for relevant results
- **Hierarchical context**: Results include ancestor headings, matched section, and direct children
- **Markdown + source code**: Indexes `.md` files and common source code extensions
- **Lower weight for code**: Source code results ranked below documentation
- **Auto-reindex**: Detects file changes and updates index on search
- **Embedding cache**: SHA256-based cache avoids re-embedding unchanged content
- **CLI + Python API**: Use from command line or embed in your application
- **Local only**: No external services, runs completely offline after model download

## Quick Links

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Configuration](configuration.md)
- [CLI Reference](cli.md)
- [Python API](python-api.md)

## How It Works

docsearch parses Markdown files into a heading hierarchy, creates semantic embeddings for each section using BGE-small-en-v1.5, and stores them in a local SQLite + NumPy index. Search queries are embedded and matched against the index using cosine similarity with optional FTS5 lexical boost.

Results are returned with full markdown context: the matched heading, its ancestors' toplevel content, and its direct children's toplevel content — so you see the relevant section in its hierarchical context.

## Architecture

```
docs/                    → Markdown + source files
    index/               → SQLite + embeddings.npy + cache/
        index.sqlite     → Files, nodes, FTS5 index
        embeddings.npy   → Normalized float32 embeddings
        cache/           → Per-node embedding cache (SHA256)
```

## Requirements

- Python 3.10+
- uv (recommended) or pip
- ~50MB for BGE-small-en-v1.5 model (downloaded on first use)