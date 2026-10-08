"""Command-line interface for docsearch."""

import logging
import os
import sys
from pathlib import Path

import click

# Suppress HF Hub warnings and progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress huggingface_hub and related logging
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub.utils").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)


class DocSearchGroup(click.Group):
    """Custom group that handles search queries and subcommands correctly."""
    
    def resolve_command(self, ctx, args):
        # args here are the remaining args after group options have been parsed
        # If first arg is a known subcommand, let click handle it normally
        # If first arg is not a known subcommand and not empty, treat as search query
        if not args:
            return super().resolve_command(ctx, args)
        
        if args[0] in self.commands:
            return super().resolve_command(ctx, args)
        
        # Prepend "search" to route to search_cmd
        return super().resolve_command(ctx, ["search"] + args)


def _build_config(ctx) -> object | None:
    """Build config lazily to improve startup time."""
    from docsearch.config import Config, IndexConfig, ModelConfig, SearchConfig

    model_config = ModelConfig(
        device=None if ctx.obj.get("device") == "auto" else ctx.obj.get("device"),
        batch_size=ctx.obj.get("batch_size", 128),
        cache_dir=ctx.obj.get("cache_dir"),
    )
    config = Config(model=model_config)
    config.index = IndexConfig(path=Path(ctx.obj["index_path"]))
    config.search = SearchConfig()
    return config


@click.group(cls=DocSearchGroup, invoke_without_command=True)
@click.option("--index", "-i", "index_path", type=click.Path(path_type=Path), default="docs.index",
              help="Index directory path")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto",
              help="Device for embedding model")
@click.option("--batch-size", type=int, default=128, help="Batch size for embedding")
@click.option("--cache-dir", type=click.Path(path_type=Path), help="Model cache directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, index_path, device, batch_size, cache_dir, verbose):
    """docsearch - Fast local semantic search for Markdown documentation.

    QUERY: Run a search query directly.

    \b
    docsearch "how does fork work?"
    docsearch index ./docs
    docsearch search "how does exec work?"
    """
    ctx.ensure_object(dict)
    ctx.obj["index_path"] = index_path
    ctx.obj["device"] = device
    ctx.obj["batch_size"] = batch_size
    ctx.obj["cache_dir"] = cache_dir
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("docs_path", type=click.Path(exists=True, path_type=Path))
@click.option("--force", "-f", is_flag=True, help="Force rebuild from scratch")
@click.pass_context
def index(ctx, docs_path, force):
    """Index a documentation directory."""
    config = _build_config(ctx)
    config.index.path = Path(ctx.obj["index_path"])

    if ctx.obj["verbose"]:
        click.echo(f"Indexing {docs_path} -> {config.index.path}")

    from docsearch.index import build_index
    metadata = build_index(docs_path, config.index.path, config, force_rebuild=force)

    click.echo(f"Indexed {metadata.file_count} files, {metadata.node_count} nodes")
    click.echo(f"Model: {metadata.model_name} (dim={metadata.embedding_dim})")


@cli.command()
@click.argument("docs_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def rebuild(ctx, docs_path):
    """Rebuild index from scratch (same as index --force)."""
    config = _build_config(ctx)
    config.index.path = Path(ctx.obj["index_path"])

    if ctx.obj["verbose"]:
        click.echo(f"Rebuilding index from {docs_path}")

    from docsearch.index import build_index
    metadata = build_index(docs_path, config.index.path, config, force_rebuild=True)

    click.echo(f"Rebuilt index: {metadata.file_count} files, {metadata.node_count} nodes")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", type=int, default=5, help="Number of results")
@click.option("--top-files", type=int, default=10, help="Files to consider at file level")
@click.option("--top-nodes", type=int, default=50, help="Nodes to consider at node level")
@click.option("--min-score", type=float, default=0.0, help="Minimum similarity score")
@click.option("--lexical-boost", type=float, default=0.1, help="Lexical (FTS) boost factor")
@click.option("--context", type=click.Choice(["parent_heading", "full_ancestry", "none"]),
              default="parent_heading", help="Context expansion policy")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search_cmd(ctx, query, top_k, top_files, top_nodes, min_score, lexical_boost, context, output_json):  # noqa: E501
    """Search the index."""
    config = _build_config(ctx)
    config.index.path = Path(ctx.obj["index_path"])
    config.search.top_k = top_k
    config.search.top_files = top_files
    config.search.top_nodes = top_nodes
    config.search.min_score = min_score
    config.search.lexical_boost = lexical_boost
    config.search.context_expansion = context

    if ctx.obj["verbose"]:
        click.echo(f"Searching: {query}")

    try:
        from docsearch.search import search as do_search
        results = do_search(query, ctx.obj["index_path"], config, top_k=top_k)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output_json:
        import json
        click.echo(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        if not results:
            click.echo("No results found")
            return

        for result in results:
            click.echo(f"{result.file}:{result.start_line}-{result.end_line}")
            click.echo(f"  {result.heading_str}")
            if result.markdown_context:
                # Show the actual markdown content
                lines = result.markdown_context.splitlines()
                snippet = "\n".join(lines)
                click.echo(snippet)
            click.echo(f"  score={result.score:.4f}")
            if ctx.obj["verbose"]:
                click.echo(f"  matched: {result.matched_node_id} (level={result.matched_level})")
            click.echo()


@cli.command()
@click.pass_context
def info(ctx):
    """Show index information."""
    from docsearch.storage import get_connection, load_metadata

    metadata = load_metadata(Path(ctx.obj["index_path"]))
    if metadata is None:
        click.echo("No index found", err=True)
        sys.exit(1)

    click.echo(f"Index: {ctx.obj['index_path']}")
    click.echo(f"Version: {metadata.version}")
    click.echo(f"Model: {metadata.model_name}")
    click.echo(f"Embedding dim: {metadata.embedding_dim}")
    click.echo(f"Files: {metadata.file_count}")
    click.echo(f"Nodes: {metadata.node_count}")

    if ctx.obj["verbose"]:
        with get_connection(Path(ctx.obj["index_path"])) as conn:
            files = conn.execute("SELECT file, child_count FROM files ORDER BY file").fetchall()
            for f in files:
                click.echo(f"  {f['file']} ({f['child_count']} nodes)")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()