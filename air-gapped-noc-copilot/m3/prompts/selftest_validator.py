"""Self-test the schema validator against the few-shot examples.

Exits non-zero if any example fails. Use in CI or as a smoke test before
deploying a new prompt / schema version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from m3.prompts.schema_validator import validate_response, CopilotUnavailable


def main() -> int:
    examples_path = HERE / "few_shot_examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for ex in examples:
        raw = json.dumps(ex["response"])
        try:
            validate_response(raw, evidence_chunks=ex.get("evidence", []))
            print(f"OK   {ex['name']}")
        except CopilotUnavailable as e:
            failures.append(f"{ex['name']}: {e}")
            print(f"FAIL {ex['name']}: {e}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print(f"\nAll {len(examples)} examples validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
