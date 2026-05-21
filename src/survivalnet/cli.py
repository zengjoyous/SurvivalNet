"""Command line interface for survivalnet."""

import click


@click.group()
def main() -> None:
    """Top-level survivalnet command."""


@main.command()
def version() -> None:
    """Print package version."""
    from . import __version__

    click.echo(__version__)


if __name__ == "__main__":
    main()
