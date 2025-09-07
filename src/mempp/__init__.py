from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only for type-checkers
    from .config import MemppSystemConfig

__version__ = "0.1.0"


def _build_config() -> MemppSystemConfig:
    # Import from lightweight config module to avoid heavy deps for CLI
    from .config import MemppSystemConfig

    return MemppSystemConfig(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "dev-key"),
        pinecone_environment=os.getenv("PINECONE_ENV", "us-east-1"),
        pinecone_index_name=os.getenv("PINECONE_INDEX", "memp-dev"),
    )


def main(argv: list[str] | None = None) -> None:
    """Tiny CLI that prints environment-derived config and version."""
    parser = argparse.ArgumentParser(prog="mempp", description="Mempp procedural memory sandbox CLI")
    parser.add_argument("command", nargs="?", default="info", choices=["info"])  # keep minimal
    args = parser.parse_args(argv)

    if args.command == "info":
        cfg = _build_config()
        print("mempp CLI")
        print(f"version: {__version__}")
        print("config:")
        for k, v in asdict(cfg).items():
            if isinstance(v, list):
                v = ",".join(map(str, v))
            if k.endswith("api_key") and v:
                v = "***"
            print(f"  - {k}: {v}")


__all__ = ["main", "__version__"]
