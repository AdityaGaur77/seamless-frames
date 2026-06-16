"""Document-class-aware chunkers for the NOC Copilot RAG pipeline.

Three chunker classes are exported:
    - TopologyChunker   one chunk per logical entity in a YAML/JSON topology file
    - RunbookChunker    markdown-aware structural splitter for operator runbooks
    - IncidentChunker   section-aware splitter for incident postmortems

All chunkers emit (chunk_id, text, metadata) tuples. The text is later
prefixed with a deterministic natural-language summary header (see
`enrich_chunk_header`) before being embedded.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


_HEADER_TEMPLATES = {
    "topology": "[Topology entity | {entity_key}={entity_id} | site={site} | role={role} | vrf={vpn} | tunnel_id={tunnel_id} | topoversion={topology_version}]",
    "runbook": "[Runbook {runbook_id} \u2014 {title} | protocol={protocol} | severity={severity} | last_reviewed={last_reviewed}]",
    "incident_detail": "[Incident {incident_id} \u2014 {section} | date={date} | sites={affected_sites} | signals={signals}]",
    "incident_summary": "[Incident {incident_id} \u2014 SUMMARY | date={date} | root_cause_class={root_cause_class}]",
}


_SINGULAR_ENTITY = {
    "devices": "device", "device": "device", "nodes": "node", "node": "node",
    "links": "link", "links_": "link", "link": "link",
    "vpns": "vpn", "vpn": "vpn", "vrfs": "vrf", "vrf": "vrf",
    "tunnels": "tunnel", "tunnel": "tunnel",
    "prefix_lists": "prefix_list", "prefix_list": "prefix_list",
    "route_maps": "route_map", "route_map": "route_map",
    "policies": "policy", "policy": "policy",
}


def _strip_frontmatter_yaml(text: str) -> tuple[str, dict]:
    if not text.startswith("---"):
        return text, {}
    end = text.find("\n---", 3)
    if end < 0:
        return text, {}
    fm = yaml.safe_load(text[3:end]) or {}
    if not isinstance(fm, dict):
        fm = {}
    return text[end + 4:].lstrip("\n"), fm


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def _token_len(s: str) -> int:
    return max(1, len(s.split()))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def enrich_chunk_header(chunk: Chunk, kind: str) -> Chunk:
    tpl = _HEADER_TEMPLATES.get(kind)
    if not tpl:
        return chunk
    meta = chunk.metadata
    safe = {k: (v if isinstance(v, (str, int, float)) else ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)) for k, v in meta.items()}
    try:
        header = tpl.format(**safe)
    except KeyError:
        header = f"[{kind} | {safe}]"
    chunk.text = f"{header}\n{chunk.text}"
    return chunk


class TopologyChunker:
    """Splits a YAML/JSON topology file into one chunk per logical entity."""

    ENTITY_KEYS = (
        "devices", "device", "nodes", "node",
        "links", "links_", "link",
        "vpns", "vpn", "vrfs", "vrf",
        "tunnels", "tunnel",
        "prefix_lists", "prefix_list",
        "route_maps", "route_map",
        "policies", "policy",
    )

    def __init__(self, target_tokens: int = 140, overlap_tokens: int = 0):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_file(self, path: Path, corpus_root: Path) -> List[Chunk]:
        rel = str(path.relative_to(corpus_root))
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            body, frontmatter = _strip_frontmatter_yaml(text)
            data = yaml.safe_load(body) or {}
        else:
            data = json.loads(text)
            frontmatter = {}
        if not isinstance(data, dict):
            return []

        top_meta = {
            "topology_version": frontmatter.get("version", data.get("version", "unknown")),
            "doc_type": "topology",
        }
        chunks: List[Chunk] = []

        for entity_key in self.ENTITY_KEYS:
            if entity_key not in data:
                continue
            items = data[entity_key]
            if isinstance(items, dict):
                items = [items]
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("id") or item.get("name") or f"{entity_key}-{idx}"
                summary = self._summarise_entity(entity_key, item)
                singular = _SINGULAR_ENTITY.get(entity_key, entity_key[:-1] if entity_key.endswith("s") else entity_key)
                body = yaml.safe_dump({singular: item}, sort_keys=False, allow_unicode=True)
                cid = _stable_id("topology", rel, entity_key, str(entity_id))
                meta = {
                    **top_meta,
                    "entity_key": entity_key,
                    "entity_id": str(entity_id),
                    "site": item.get("site", "n/a"),
                    "role": item.get("role", "n/a"),
                    "vpn": ",".join(item.get("vrfs", []) or []) or item.get("vpn", "n/a"),
                    "tunnel_id": item.get("tunnel_id", "n/a"),
                    "asn": str(item.get("asn", "n/a")),
                    "title": str(item.get("name") or item.get("id") or entity_id),
                }
                chunks.append(Chunk(cid, f"{summary}\n\n{body}", meta))
        return chunks

    @staticmethod
    def _summarise_entity(key: str, item: Dict[str, Any]) -> str:
        name = item.get("name") or item.get("id") or "entity"
        singular = _SINGULAR_ENTITY.get(key, key[:-1] if key.endswith("s") else key)
        role = item.get("role")
        site = item.get("site")
        desc = item.get("description")
        bits = [f"{singular}: {name}"]
        if role:
            bits.append(f"role={role}")
        if site:
            bits.append(f"site={site}")
        if desc:
            bits.append(desc)
        return " | ".join(bits)


class RunbookChunker:
    """Markdown structural splitter for runbooks."""

    HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)
    STEP_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)

    def __init__(self, target_tokens: int = 350, overlap_tokens: int = 50):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_file(self, path: Path, corpus_root: Path) -> List[Chunk]:
        rel = str(path.relative_to(corpus_root))
        text = path.read_text(encoding="utf-8")
        runbook_id, top_meta = self._extract_frontmatter(text)
        body = self._strip_frontmatter(text)

        sections: List[Tuple[str, str]] = []
        last_pos = 0
        last_heading = ""
        for m in self.HEADING_RE.finditer(body):
            if last_pos == 0 and m.start() > 0:
                pre = body[:m.start()].strip()
                if pre:
                    sections.append(("", pre))
            if last_pos > 0:
                sections.append((last_heading, body[last_pos:m.start()].strip()))
            last_heading = f"{len(m.group(1))}#{m.group(2).strip()}"
            last_pos = m.end()
        if last_pos < len(body):
            sections.append((last_heading, body[last_pos:].strip()))

        merged: List[Tuple[str, str]] = []
        buf_heading = ""
        buf_body: List[str] = []
        buf_tokens = 0
        for heading, content in sections:
            t = _token_len(content)
            if buf_tokens + t > self.target_tokens and buf_body:
                merged.append((buf_heading, "\n\n".join(buf_body).strip()))
                carry = " ".join(buf_body[-1].split()[-self.overlap_tokens:]) if self.overlap_tokens else ""
                buf_body = [carry] if carry else []
                buf_tokens = _token_len(carry) if carry else 0
                buf_heading = heading
            if not buf_body:
                buf_heading = heading or buf_heading
            buf_body.append(content)
            buf_tokens += t
        if buf_body:
            merged.append((buf_heading, "\n\n".join(buf_body).strip()))

        chunks: List[Chunk] = []
        for idx, (heading, content) in enumerate(merged):
            cid = _stable_id("runbook", rel, str(idx), content[:80])
            meta = {
                **top_meta,
                "runbook_id": runbook_id,
                "doc_type": "runbook",
                "section_heading": heading,
                "protocol": top_meta.get("protocol", "general"),
                "severity": top_meta.get("severity", "P3"),
                "last_reviewed": top_meta.get("last_reviewed", "unknown"),
                "steps": self._extract_steps(content),
            }
            chunks.append(Chunk(cid, content, meta))
        return chunks

    @staticmethod
    def _extract_frontmatter(text: str) -> Tuple[str, Dict[str, Any]]:
        if not text.startswith("---"):
            return ("RB-UNKNOWN", {})
        end = text.find("\n---", 3)
        if end < 0:
            return ("RB-UNKNOWN", {})
        fm = yaml.safe_load(text[3:end]) or {}
        return (str(fm.get("id", "RB-UNKNOWN")), fm)

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        if not text.startswith("---"):
            return text
        end = text.find("\n---", 3)
        if end < 0:
            return text
        return text[end + 4:].lstrip("\n")

    @staticmethod
    def _extract_steps(body: str) -> List[str]:
        return [m.group(0).strip() for m in RunbookChunker.STEP_RE.finditer(body)][:20]


class IncidentChunker:
    """Section-aware splitter for incident postmortems."""

    SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

    def __init__(self, detail_target_tokens: int = 300, summary_target_tokens: int = 80, overlap_tokens: int = 0):
        self.detail_target_tokens = detail_target_tokens
        self.summary_target_tokens = summary_target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_file(self, path: Path, corpus_root: Path) -> List[Chunk]:
        rel = str(path.relative_to(corpus_root))
        text = path.read_text(encoding="utf-8")
        incident_id, top_meta = self._extract_frontmatter(text)
        body = self._strip_frontmatter(text)

        sections: List[Tuple[str, str]] = []
        last_pos = 0
        last_title = ""
        for m in self.SECTION_RE.finditer(body):
            if last_pos > 0:
                sections.append((last_title, body[last_pos:m.start()].strip()))
            last_title = m.group(1).strip()
            last_pos = m.end()
        if last_pos < len(body):
            sections.append((last_title, body[last_pos:].strip()))

        chunks: List[Chunk] = []
        summary_text = self._make_summary(sections, top_meta)
        if summary_text:
            cid = _stable_id("incident", rel, "summary", summary_text[:60])
            chunks.append(Chunk(
                cid,
                summary_text,
                {**top_meta, "incident_id": incident_id, "doc_type": "incident_summary", "section": "SUMMARY"},
            ))

        for section_name, content in sections:
            if not content:
                continue
            if _token_len(content) <= self.detail_target_tokens:
                cid = _stable_id("incident", rel, section_name, content[:60])
                chunks.append(Chunk(
                    cid,
                    content,
                    {**top_meta, "incident_id": incident_id, "doc_type": "incident_detail", "section": section_name},
                ))
            else:
                for sub in self._split_long(content, self.detail_target_tokens, self.overlap_tokens):
                    cid = _stable_id("incident", rel, section_name, sub[:60])
                    chunks.append(Chunk(
                        cid,
                        sub,
                        {**top_meta, "incident_id": incident_id, "doc_type": "incident_detail", "section": section_name},
                    ))
        return chunks

    @staticmethod
    def _extract_frontmatter(text: str) -> Tuple[str, Dict[str, Any]]:
        if not text.startswith("---"):
            return ("INC-UNKNOWN", {})
        end = text.find("\n---", 3)
        if end < 0:
            return ("INC-UNKNOWN", {})
        fm = yaml.safe_load(text[3:end]) or {}
        return (str(fm.get("id", "INC-UNKNOWN")), fm)

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        if not text.startswith("---"):
            return text
        end = text.find("\n---", 3)
        if end < 0:
            return text
        return text[end + 4:].lstrip("\n")

    @staticmethod
    def _split_long(text: str, target: int, overlap: int) -> List[str]:
        words = text.split()
        out: List[str] = []
        i = 0
        step = max(1, target - overlap)
        while i < len(words):
            out.append(" ".join(words[i:i + target]))
            if i + target >= len(words):
                break
            i += step
        return out

    @staticmethod
    def _make_summary(sections: List[Tuple[str, str]], meta: Dict[str, Any]) -> str:
        sym = next((c for h, c in sections if h.lower() == "symptom"), "")
        rc = next((c for h, c in sections if h.lower() in ("root cause", "root_cause", "rootcause")), "")
        rem = next((c for h, c in sections if h.lower() in ("remediation", "fix", "resolution")), "")
        rcc = meta.get("root_cause_class", "unknown")
        return f"Root cause class: {rcc}. Symptom: {sym[:240]}. Root cause: {rc[:240]}. Remediation: {rem[:240]}"


def get_chunker_for(path: Path) -> Optional[Any]:
    parts = [p.lower() for p in path.parts]
    if "topology" in parts:
        return TopologyChunker()
    if "runbooks" in parts:
        return RunbookChunker()
    if "incidents" in parts:
        return IncidentChunker()
    return None


def chunk_corpus(corpus_root: Path) -> Iterable[Chunk]:
    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file():
            continue
        chunker = get_chunker_for(path)
        if chunker is None:
            continue
        for chunk in chunker.chunk_file(path, corpus_root):
            kind = (
                "topology" if "topology" in str(path).lower()
                else "runbook" if "runbooks" in str(path).lower()
                else "incident_detail" if "incident" in str(path).lower()
                else "other"
            )
            yield enrich_chunk_header(chunk, kind)
