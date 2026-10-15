# Quickstart

Get up and running with docsearch in minutes.

## 1. Index your documentation

```bash
docsearch index ./my-docs
```

This will:

- Parse all `.md` and source code files
- Create embeddings using BGE-small-en-v1.5
- Store the index in `./docs.index` (default)

## 2. Search your docs

```bash
docsearch "search query here"
```

Results show:

- File path and line numbers
- Heading hierarchy (e.g., `H1 > H2 > H3`)
- Full markdown context under the matched heading
- Relevance score (0.0 to 1.0)

## 3. Rebuild when docs change

```bash
docsearch rebuild ./my-docs
```

Or use auto-reindex: the index automatically checks for file changes on each search (configurable).

## 4. Get help

```bash
docsearch --help
# or
docsearch <command> --help
```
