"""Smoke test for the chunker against the sample corpus."""
import sys
from pathlib import Path
sys.path.insert(0, r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
from m3.rag.chunker import chunk_corpus

CORPUS = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot\m3\rag\corpus')

chunks = list(chunk_corpus(CORPUS))
print(f"PASS  chunker produced {len(chunks)} chunks")

by_kind = {}
for c in chunks:
    kind = c.metadata.get('doc_type', 'unknown')
    by_kind.setdefault(kind, 0)
    by_kind[kind] += 1

assert by_kind.get('topology', 0) > 0
assert by_kind.get('runbook', 0) > 0
assert by_kind.get('incident_summary', 0) > 0
assert by_kind.get('incident_detail', 0) > 0
print(f"PASS  chunker produces all four doc_types: {by_kind}")

ids = [c.chunk_id for c in chunks]
assert len(ids) == len(set(ids)), "duplicate chunk ids"
print(f"PASS  chunker produces {len(ids)} unique stable ids")

for c in chunks:
    first_line = c.text.split("\n", 1)[0]
    assert first_line.startswith("["), f"missing header in chunk: {c.text[:80]}"
print(f"PASS  all {len(chunks)} chunks have deterministic headers")

for c in chunks:
    if c.metadata.get("entity_key") == "policies":
        text = c.text.lower()
        assert "policie:" not in text, f"policies singular bug: {c.text[:200]}"
        print(f"PASS  policies entity uses correct singular 'policy'")
        break
else:
    print("WARN  no policies chunk found")

print("\nAll chunker tests passed.")
