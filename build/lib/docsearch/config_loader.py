"""Load and save docsearch configuration from docs.index/config.json."""

import fnmatch
import json
from pathlib import Path

from docsearch.config import Config

CONFIG_FILE = "config.json"


def _glob_match(text: str, pattern: str) -> bool:
    """Match text against a glob pattern, handling `**/` prefix."""
    if fnmatch.fnmatchcase(text, pattern):
        return True
    # Handle ** prefix: strip leading "**/ " so pattern matches basename too
    if pattern.startswith("**/"):
        stripped = pattern[3:]
        if fnmatch.fnmatchcase(text, stripped):
            return True
    return False


def load_config(index_path: Path | str) -> Config:
    """Load config from docs.index/config.json if it exists, else return defaults."""
    index_path = Path(index_path)
    config_path = index_path / CONFIG_FILE
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        return Config.from_dict(data)
    return Config()


def save_config(config: Config, index_path: Path | str) -> None:
    """Save config to docs.index/config.json."""
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    config_path = index_path / CONFIG_FILE
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def should_index_file(filepath: Path, docs_root: Path, config: Config) -> bool:
    """Check if a file should be indexed based on globs and ignore_dirs."""
    try:
        rel = filepath.relative_to(docs_root)
    except ValueError:
        return False  # Outside docs root

    name = rel.name
    rel_str = str(rel)
    parts = list(rel.parts)

    # Check ignore_dirs
    for part in parts:
        if part in config.ignore_dirs:
            return False

    # Check ignored_globs: if any pattern matches, don't index
    for pattern in config.ignored_globs:
        if _glob_match(rel_str, pattern) or _glob_match(name, pattern):
            return False
        for part in parts:
            if _glob_match(part, pattern):
                return False

    # Check allowed_globs: if empty, allow all; otherwise must match at least one
    allowed = config.allowed_globs
    if allowed:
        matches = (
            any(_glob_match(rel_str, g) for g in allowed)
            or any(_glob_match(name, g) for g in allowed)
            or any(_glob_match(p, g) for p in parts for g in allowed)
        )
        if not matches:
            return False
    return True