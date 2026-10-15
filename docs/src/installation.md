# Installation

## Prerequisites

- Python >= 3.10
- uv or pip
- Git (optional, for source code indexing)

## Install with uv

```bash
uv tool install docsearch
```

## Install with pip

```bash
pip install docsearch
```

## From Source

```bash
git clone https://github.com/Omena0/docsearch.git
cd docsearch
uv sync  # or: pip install -e ".[all]"
```

## Post-Install

The first search will automatically download the BGE-small-en-v1.5 embedding model (~50 MB). This only happens once.

## Verify Installation

```bash
docsearch --version
```

Expected output: `docsearch <version>`

## Quickstart

We recommend using [Fr-docs](https://omena0.dev/fr-docs/) to simplify the
deployment of your documentation site.

```bash
# 1. Create some docs
mkdir -p my-docs
echo "# Hello World

## Introduction

This is a sample documentation file." > my-docs/hello.md

# 2. Index the docs
docsearch index my-docs

# 3. Search
docsearch "Hello World"
```
