# Configuration

docsearch can be configured via CLI options, environment variables, or Python API.

## CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--index`, `-i` | `-i` | Index directory path | `./docs.index` |
| `--device` | | Device: `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--batch-size` | | Embedding batch size | `128` |
| `--cache-dir` | | Model cache directory | `~/.cache/huggingface` |
| `--verbose`, `-v` | `-v` | Verbose output | `false` |
| `--version` | | Show version and exit | |

## Search Options

| Option | Description | Default |
|--------|-------------|---------|
| `--top-k`, `-k` | Number of results | `5` |
| `--top-files` | Files to consider at file level | `10` |
| `--top-nodes` | Nodes to consider at node level | `50` |
| `--min-score` | Minimum similarity score | `0.0` |
| `--lexical-boost` | FTS lexical boost factor | `0.1` |
| `--context` | Context: `parent_heading`, `full_ancestry`, `none` | `parent_heading` |
| `--json` | Output as JSON | `false` |

## Python Configuration

```python
from docsearch.config import Config, ChunkingConfig, IndexConfig, ModelConfig, SearchConfig

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
    # Directories to ignore during indexing
    ignore_dirs=(".git", "node_modules", "__pycache__", ".venv", "venv"),
    # Markdown file extensions
    markdown_extensions=(".md", ".markdown"),
    # Source code extensions (lower weight)
    source_extensions=(
        ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
        ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
        ".sh", ".sql", ".toml", ".yaml", ".yml", ".json",
    ),
    # Weight for source code results (0.0-1.0)
    source_weight=0.3,
    # Auto reindex on search
    auto_reindex=True,
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_HUB_DISABLE_PROGRESS_BARS` | Disable HF Hub progress bars |
| `HF_HUB_DISABLE_TELEMETRY` | Disable HF Hub telemetry |
| `TRANSFORMERS_VERBOSITY` | Transformers logging level |
| `TOKENIZERS_PARALLELISM` | Tokenizer parallelism (true/false) |

## Indexing Configuration

- `ignore_dirs`: Directories to skip during indexing
- `markdown_extensions`: File extensions treated as Markdown (full weight)
- `source_extensions`: File extensions treated as source code (lower weight)
- `source_weight`: Weight multiplier for source code results (default 0.3)

## Search Tuning

- `top_k`: Final number of results returned
- `top_files`: Files to score at file level (broad search)
- `top_nodes`: Nodes to score within top files (detailed search)
- `lexical_boost`: Weight for FTS5 exact match boost (0 to disable)
- `context_expansion`: How much context to include in line ranges
  - `parent_heading`: Expand to parent heading start
  - `full_ancestry`: Expand to document root
  - `none`: No expansion