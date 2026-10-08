# Configuration

## Configuration File

Create a `config.json` in your docs directory:

```json
{
  "project_name": "My Project",
  "project_url": "https://github.com/user/repo",
  "site_path_prefix": "/",
  "src_dir": "src",
  "out_dir": "site",
  "docs_dir": ".",
  "sidebar": [
    ["Getting Started", [
      ["index", "Home"],
      ["installation", "Installation"]
    ]]
  ],
  "source_files": {
    "search_dirs": ["parent", "src_parent", "docs"],
    "patterns": [
      "docs/src/*.md",
      "docs/*.js",
      "docs/*.css"
    ],
    "ignore_dirs": []
  }
}
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `project_name` | string | "Project Name" | Display name for your project |
| `project_url` | string | "" | Project repository URL |
| `site_path_prefix` | string | "/" | URL path prefix for deployment |
| `src_dir` | string | "src" | Source markdown directory |
| `out_dir` | string | "site" | Output HTML directory |
| `docs_dir` | string | "." | Docs root directory |
| `sidebar` | array | [] | Navigation structure |
| `source_files.patterns` | array | ["docs/src/*.md", "docs/*.js", "docs/*.css"] | Glob patterns for source files (see note below) |
| `source_files.ignore_dirs` | array | [] | Directory names to skip |

## Build Options

```json
{
  "build": {
    "workers": 6,
    "minify_html": true,
    "optimize_html": true,
    "zstd_level": 22,
    "search_index_filename": "search_index.zst",
    "git_meta_filename": "git_meta.json"
  }
}
```

## Feature Toggles

```json
{
  "features": {
    "backlinks": true,
    "related": true,
    "auto_link": true,
    "versioning": true,
    "search": true,
    "code_references": true,
    "link_preview": true,
    "code_highlighting": true,
    "blockquotes": true,
    "ext_tags": true
  }
}
```

## Versioning

```json
{
  "versioning": {
    "commit_message_pattern": "^\\s*([0-9]+[A-Za-z])\\s*[-:—–]\\s*(.+)",
    "live_label": "Live"
  }
}
```

## Source Files

The `source_files` configuration controls which files are indexed for code references and search:

- `patterns`: Glob patterns (relative to each search directory). Use `*` for any filename, `**` for recursive.
- `search_dirs`: Where to search. Options: `"parent"` (project root), `"src_parent"`, `"docs"`.
- `ignore_dirs`: Directory names to skip during search.

Add your own patterns to `patterns` to include additional file types or directories.
