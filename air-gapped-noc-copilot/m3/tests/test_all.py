"""Final comprehensive smoke test that runs every test file in sequence.

Run from the repo root:
    python -m m3.tests.test_all

Exits 0 if all tests pass, non-zero otherwise.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
sys.path.insert(0, str(ROOT))
TESTS_DIR = ROOT / "m3" / "tests"

print("=" * 70)
print("Air-Gapped NOC Copilot - Final Review Smoke Test")
print("=" * 70)

test_files = [
    ("chunker", TESTS_DIR / "test_chunker.py"),
    ("RAG e2e (in-process)", TESTS_DIR / "test_rag_e2e.py"),
    ("RAG CLI", TESTS_DIR / "test_rag_cli.py"),
    ("prompts", TESTS_DIR / "test_prompts.py"),
    ("validator", TESTS_DIR / "test_validator.py"),
    ("playbooks", TESTS_DIR / "test_playbooks.py"),
    ("telemetry", TESTS_DIR / "test_telemetry.py"),
    ("validator self-test (bundled)", None),
]

ENV_OVERRIDES = {
    "PYTHONPATH": str(ROOT),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "NOC_COPILOT_FAKE_EMBEDDER": "1",
    "NOC_COPILOT_SKIP_AIRGAP_CHECK": "1",
}
env = {**os.environ, **ENV_OVERRIDES}

failed = 0
passed = 0
ok_count = 0

for name, path in test_files:
    print(f"\n--- {name} ---")
    if path is None:
        cmd = [sys.executable, "-m", "m3.prompts.selftest_validator"]
    else:
        if not path.exists():
            print(f"  SKIP  file not found: {path}")
            continue
        cmd = [sys.executable, str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"  FAIL  timeout")
        failed += 1
        continue
    pass_lines = [ln for ln in res.stdout.splitlines() if ln.startswith("PASS")]
    fail_lines = [ln for ln in res.stdout.splitlines() if ln.startswith("FAIL")]
    ok_lines = [ln for ln in res.stdout.splitlines() if ln.startswith("OK")]
    for ln in pass_lines:
        print(f"  {ln.strip()}")
    for ln in ok_lines:
        print(f"  {ln.strip()}")
    for ln in fail_lines:
        print(f"  {ln.strip()}")
    if res.returncode == 0:
        passed += len(pass_lines)
        ok_count += len(ok_lines)
    else:
        failed += max(1, len(fail_lines) or 1)
        if res.stderr:
            print(f"  STDERR (tail): {res.stderr[-400:]}")

print()
print("=" * 70)
print(f"Smoke test complete: {passed + ok_count} passed, {failed} failed")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)
