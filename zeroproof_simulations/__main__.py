"""CLI. Rank a JSONL file: python -m zeroproof_simulations path.jsonl"""
from .quality import main

if __name__ == "__main__":
    raise SystemExit(main())
