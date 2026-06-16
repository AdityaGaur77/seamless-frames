"""Offline LLM HTTP client — production-grade for air-gapped deployment.

Two backends are supported:

1. **OpenAI-compatible** (llama.cpp ``/server``, vLLM, Ollama OpenAI-compat):
   ``/v1/chat/completions`` — used when ``api_backend == "openai"``.

2. **Ollama native** (``http://localhost:11434``):
   ``/api/chat`` (preferred, multi-turn) or ``/api/generate`` (single-turn).
   Used when ``api_backend == "ollama"``.

Both backends are *synchronous* and *fail-loud*:

* configurable ``max_retries`` (default 1),
* no streaming,
* on non-200 / connection failure, raises
  :class:`OfflineLLMUnavailable` with the raw status + body,
* no proxy / HTTPS redirect — all traffic stays on loopback.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

LOG = logging.getLogger("controller.llm")


class OfflineLLMUnavailable(RuntimeError):
    """Raised when the offline LLM is unreachable or returns an error."""

    def __init__(self, reason: str, envelope: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.envelope = envelope or {"copilot_unavailable": True, "reason": reason}


# ─────────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _BaseConfig:
    base_url: str
    model_name: str
    request_timeout_seconds: int
    max_retries: int
    temperature: float
    top_p: float
    max_tokens: int


# ─────────────────────────────────────────────────────────────────────
# OpenAI-compatible backend (llama.cpp / vLLM / Ollama-compat)
# ─────────────────────────────────────────────────────────────────────


class OfflineLLMClient:
    """Synchronous chat-completion client for the local inference server.

    Talks the OpenAI-compatible ``/v1/chat/completions`` endpoint.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        api_path: str = "/v1/chat/completions",
        model_name: str = "airgap-noc",
        request_timeout_seconds: int = 60,
        max_retries: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ):
        self._cfg = _BaseConfig(
            base_url=base_url.rstrip("/"),
            model_name=model_name,
            request_timeout_seconds=int(request_timeout_seconds),
            max_retries=int(max(max_retries, 1)),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        self._api_path = api_path if api_path.startswith("/") else "/" + api_path
        self._session = requests.Session()
        self._session.trust_env = False

    def health(self) -> bool:
        url = self._cfg.base_url + "/health"
        try:
            r = self._session.get(url, timeout=min(5, self._cfg.request_timeout_seconds))
            return r.status_code == 200
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not messages:
            raise OfflineLLMUnavailable("empty_messages")

        url = self._cfg.base_url + self._api_path
        body: Dict[str, Any] = {
            "model": self._cfg.model_name,
            "messages": messages,
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
            "max_tokens": self._cfg.max_tokens,
            "stream": False,
        }
        if response_format:
            body["response_format"] = response_format

        return self._post_json(url, body, extract_fn=self._extract_chat)

    def _post_json(
        self,
        url: str,
        body: Dict[str, Any],
        *,
        extract_fn,
    ) -> str:
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < self._cfg.max_retries:
            attempts += 1
            t0 = time.monotonic()
            try:
                r = self._session.post(
                    url,
                    data=json.dumps(body),
                    headers={"Content-Type": "application/json"},
                    timeout=self._cfg.request_timeout_seconds,
                )
            except requests.ConnectionError as exc:
                last_exc = exc
                LOG.warning(
                    "LLM connection refused attempt=%d url=%s: %s",
                    attempts, url, exc,
                )
                continue
            except requests.Timeout as exc:
                last_exc = exc
                LOG.warning(
                    "LLM timeout attempt=%d elapsed_ms=%d: %s",
                    attempts,
                    int((time.monotonic() - t0) * 1000),
                    exc,
                )
                continue
            except requests.RequestException as exc:
                last_exc = exc
                LOG.warning("LLM request error attempt=%d: %s", attempts, exc)
                continue

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            LOG.info("LLM response status=%d elapsed_ms=%d", r.status_code, elapsed_ms)

            if r.status_code != 200:
                raise OfflineLLMUnavailable(
                    reason=f"http_{r.status_code}",
                    envelope={
                        "copilot_unavailable": True,
                        "reason": f"http_{r.status_code}",
                        "status_code": r.status_code,
                        "body_excerpt": (r.text or "")[:400],
                    },
                )
            try:
                payload = r.json()
            except json.JSONDecodeError as exc:
                raise OfflineLLMUnavailable(
                    reason="non_json_response",
                    envelope={
                        "copilot_unavailable": True,
                        "reason": "non_json_response",
                        "body_excerpt": (r.text or "")[:400],
                    },
                ) from exc

            content = extract_fn(payload)
            if not content:
                raise OfflineLLMUnavailable(
                    reason="empty_completion",
                    envelope={"copilot_unavailable": True, "reason": "empty_completion"},
                )
            return content

        raise OfflineLLMUnavailable(
            reason="max_retries_exceeded",
            envelope={
                "copilot_unavailable": True,
                "reason": "max_retries_exceeded",
                "last_error": repr(last_exc) if last_exc else None,
            },
        )

    @staticmethod
    def _extract_chat(payload: Dict[str, Any]) -> str:
        return (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )


# ─────────────────────────────────────────────────────────────────────
# Ollama native backend (/api/chat  and  /api/generate)
# ─────────────────────────────────────────────────────────────────────


class OllamaClient:
    """Synchronous client for the Ollama native HTTP API.

    Supports both ``/api/chat`` (multi-turn, preferred) and
    ``/api/generate`` (single-turn fallback).  The client automatically
    selects ``/api/chat`` when ``messages`` has more than one entry.

    Connects to ``http://localhost:11434`` by default (air-gapped host).
    """

    _HEALTH_PATH = "/api/tags"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "llama3:latest",
        request_timeout_seconds: int = 120,
        max_retries: int = 2,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        num_ctx: int = 4096,
        num_gpu: int = -1,
    ):
        self._cfg = _BaseConfig(
            base_url=base_url.rstrip("/"),
            model_name=model_name,
            request_timeout_seconds=int(request_timeout_seconds),
            max_retries=int(max(max_retries, 1)),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        self._num_ctx = int(num_ctx)
        self._num_gpu = int(num_gpu)
        self._session = requests.Session()
        self._session.trust_env = False

    # ── health ─────────────────────────────────────────────────

    def health(self) -> bool:
        try:
            r = self._session.get(
                self._cfg.base_url + self._HEALTH_PATH,
                timeout=min(5, self._cfg.request_timeout_seconds),
            )
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            return any(m.get("name", "").startswith(self._cfg.model_name) for m in models)
        except requests.RequestException:
            return False

    # ── public API ─────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat completion.  Routes to ``/api/chat`` or
        ``/api/generate`` depending on message count.

        Raises :class:`OfflineLLMUnavailable` on any failure.
        """
        if not messages:
            raise OfflineLLMUnavailable("empty_messages")

        if len(messages) == 1:
            return self._generate(messages[0].get("content", ""), response_format)
        return self._chat(messages, response_format)

    # ── /api/chat (multi-turn) ─────────────────────────────────

    def _chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = self._cfg.base_url + "/api/chat"
        body: Dict[str, Any] = {
            "model": self._cfg.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "num_predict": self._cfg.max_tokens,
                "num_ctx": self._num_ctx,
            },
        }
        if self._num_gpu >= 0:
            body["options"]["num_gpu"] = self._num_gpu
        if response_format:
            body["format"] = response_format

        return self._post_json(url, body, extract_fn=self._extract_chat_response)

    # ── /api/generate (single-turn) ────────────────────────────

    def _generate(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = self._cfg.base_url + "/api/generate"
        body: Dict[str, Any] = {
            "model": self._cfg.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "num_predict": self._cfg.max_tokens,
                "num_ctx": self._num_ctx,
            },
        }
        if self._num_gpu >= 0:
            body["options"]["num_gpu"] = self._num_gpu
        if response_format:
            body["format"] = response_format

        return self._post_json(url, body, extract_fn=self._extract_generate_response)

    # ── shared POST + retry logic ──────────────────────────────

    def _post_json(
        self,
        url: str,
        body: Dict[str, Any],
        *,
        extract_fn,
    ) -> str:
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < self._cfg.max_retries:
            attempts += 1
            t0 = time.monotonic()
            try:
                r = self._session.post(
                    url,
                    data=json.dumps(body),
                    headers={"Content-Type": "application/json"},
                    timeout=self._cfg.request_timeout_seconds,
                )
            except requests.ConnectionError as exc:
                last_exc = exc
                LOG.warning(
                    "Ollama connection refused attempt=%d url=%s: %s",
                    attempts, url, exc,
                )
                continue
            except requests.Timeout as exc:
                last_exc = exc
                LOG.warning(
                    "Ollama timeout attempt=%d elapsed_ms=%d: %s",
                    attempts,
                    int((time.monotonic() - t0) * 1000),
                    exc,
                )
                continue
            except requests.RequestException as exc:
                last_exc = exc
                LOG.warning("Ollama request error attempt=%d: %s", attempts, exc)
                continue

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            LOG.info("Ollama response status=%d elapsed_ms=%d", r.status_code, elapsed_ms)

            if r.status_code != 200:
                raise OfflineLLMUnavailable(
                    reason=f"ollama_http_{r.status_code}",
                    envelope={
                        "copilot_unavailable": True,
                        "reason": f"ollama_http_{r.status_code}",
                        "status_code": r.status_code,
                        "body_excerpt": (r.text or "")[:400],
                    },
                )
            try:
                payload = r.json()
            except json.JSONDecodeError as exc:
                raise OfflineLLMUnavailable(
                    reason="ollama_non_json",
                    envelope={
                        "copilot_unavailable": True,
                        "reason": "ollama_non_json",
                        "body_excerpt": (r.text or "")[:400],
                    },
                ) from exc

            content = extract_fn(payload)
            if not content:
                raise OfflineLLMUnavailable(
                    reason="ollama_empty_completion",
                    envelope={"copilot_unavailable": True, "reason": "ollama_empty_completion"},
                )
            return content

        raise OfflineLLMUnavailable(
            reason="ollama_max_retries_exceeded",
            envelope={
                "copilot_unavailable": True,
                "reason": "ollama_max_retries_exceeded",
                "last_error": repr(last_exc) if last_exc else None,
            },
        )

    @staticmethod
    def _extract_chat_response(payload: Dict[str, Any]) -> str:
        return payload.get("message", {}).get("content", "")

    @staticmethod
    def _extract_generate_response(payload: Dict[str, Any]) -> str:
        return payload.get("response", "")
