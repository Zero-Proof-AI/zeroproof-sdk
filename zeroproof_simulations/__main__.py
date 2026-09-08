"""CLI. Rank a JSONL file: python -m zeroproof_simulations path.jsonl"""
from .score.quality import main

if __name__ == "__main__":
    raise SystemExit(main())
