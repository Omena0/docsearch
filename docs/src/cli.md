# CLI Reference

The `docsearch` CLI provides commands for indexing and searching your documentation.

## Commands

```bash
docsearch [OPTIONS] QUERY
docsearch search QUERY
docsearch index PATH
docsearch rebuild PATH
docsearch info
```

## Direct Search Query

The fastest way to search:

```bash
docsearch "how does fork work?"
```

If the first argument is not a known subcommand, docsearch treats it as a search query.

## Commands

### docsearch search

```bash
docsearch search QUERY [OPTIONS]
```

Search the index.

**Options:**

- `--top-k, -k INTEGER` — Number of results to return (default: 5)
- `--top-files INTEGER` — Files to consider at file level (default: 10)
- `--top-nodes INTEGER` — Nodes to consider at node level (default: 50)
- `--min-score FLOAT` — Minimum similarity score (default: 0.0)
- `--lexical-boost FLOAT` — FTS lexical boost factor (default: 0.1)
- `--context [parent_heading|full_ancestry|none]` — Context expansion (default: parent_heading)
- `--json` — Output results as JSON
- `--help` — Show this message

### docsearch index

```bash
docsearch index PATH [OPTIONS]
```

Index a documentation directory.

**Options:**

- `--force, -f` — Force rebuild from scratch
- `--help` — Show this message

### docsearch rebuild

```bash
docsearch rebuild PATH
```

Rebuild index from scratch (same as `docsearch index --force`).

### docsearch info

```bash
docsearch info
```

Show index information (file count, node count, model, etc.).

## Examples

```bash
# Search for a topic
docsearch "memory management"

# Search in a specific index
docsearch -i ./project.index "how does exec work?"

# Get more results
docsearch -k 10 "kernel processes"

# Verbose output
docsearch -v "signal handling"

# JSON output
docsearch --json "file descriptors"

# Verbose index search
docsearch search "fork" -v -k 5

# Info about current index
docsearch info
```

## Exit Codes

- `0` — Success
- `1` — Error (index not found, search error, etc.)

## Performance Tips

- Use the default index location (`./docs.index`) for consistency
- Rebuild only when docs change, or use auto-reindex on search
- Lower `--top-nodes` reduces search time for large indexes
