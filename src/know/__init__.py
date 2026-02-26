"""know - Living documentation generator for codebases."""

__version__ = "0.8.9"
__author__ = "Sushil Kumar"


def main() -> None:
    """CLI entrypoint shim with lazy import.

    Avoid importing CLI dependencies (click/rich) during package import so
    library-only consumers and benchmark tooling can import lightweight modules.
    """
    from know.cli import main as cli_main

    cli_main()

__all__ = ["main", "__version__"]
