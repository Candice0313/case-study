#!/usr/bin/env python3
"""
Placeholder: Evaluation runner.
- Load test cases (in-scope / out-of-scope)
- Call chat API and assert scope + response shape
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def main() -> None:
    print("Eval script: not yet implemented. Add test cases and API assertions.")
    # TODO: run /chat on fixtures, check scope_label and content


if __name__ == "__main__":
    main()
