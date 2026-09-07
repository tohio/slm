"""One read-only search interaction; provider details stay outside model data."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from urllib.parse import urlsplit

from config.chat import parse_tool_call, safe_json

MAX_RESULTS = 3
MAX_RESULT_CHARS = 2400
MAX_RESPONSE_BYTES = 256_000


def normalize_results(rows: list[dict]) -> list[dict]:
    results = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url", ""))
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            continue
        if len(url) > 320 or parsed.username or parsed.password:
            continue
        result = {
            "title": str(row.get("title", ""))[:120],
            "url": url,
            "snippet": str(row.get("snippet", ""))[:400],
        }
        if len(safe_json(results + [result])) > MAX_RESULT_CHARS:
            break
        results.append(result)
        if len(results) == MAX_RESULTS:
            break
    return results


def search_from_environment() -> Callable[[str], list[dict]]:
    """Choose an operator-configured adapter, never a model-provided endpoint."""
    provider = os.environ.get("WEB_SEARCH_PROVIDER", "").strip().lower()
    headers = {"Accept": "application/json"}
    if provider == "brave":
        endpoint = "https://api.search.brave.com/res/v1/web/search"
        key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not key:
            raise ValueError("BRAVE_SEARCH_API_KEY is required for Brave search")
        headers["X-Subscription-Token"] = key
    elif provider == "searxng":
        endpoint = os.environ.get("SEARXNG_URL", "")
        parsed = urlsplit(endpoint)
        if (not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment
                or (parsed.scheme != "https" and not (
                    parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                ))):
            raise ValueError("SEARXNG_URL must be an HTTPS /search endpoint (HTTP allowed on loopback)")
    else:
        raise ValueError("Set WEB_SEARCH_PROVIDER to brave or searxng before enabling web search")

    def search(query: str) -> list[dict]:
        import requests

        params = ({"q": query, "count": MAX_RESULTS} if provider == "brave"
                  else {"q": query, "format": "json"})
        deadline = time.monotonic() + 10.0
        body = bytearray()
        # No automatic retries, redirects, pagination, or subsequent page fetches.
        with requests.get(endpoint, params=params, headers=headers, stream=True,
                          timeout=(5, 1), allow_redirects=False) as response:
            if response.status_code != 200:
                raise RuntimeError("Search request failed")
            # Inspect each streamed byte so a slow drip cannot hide within a
            # large buffered chunk and bypass the wall-time check.
            for chunk in response.iter_content(chunk_size=1):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES or time.monotonic() > deadline:
                    raise RuntimeError("Search response exceeded its byte/time budget")
        data = json.loads(body)
        rows = data.get("web", {}).get("results", []) if provider == "brave" else data.get("results", [])
        if not isinstance(rows, list):
            raise ValueError("Invalid search results")
        return normalize_results([
            {"title": row.get("title", ""), "url": row.get("url", ""),
             "snippet": row.get("description" if provider == "brave" else "content", "")}
            for row in rows if isinstance(row, dict)
        ])

    return search


def run_tool_turn(generate: Callable, messages: list[dict], search: Callable) -> list[dict]:
    """Return new messages without mutating history; execute at most one search."""
    text = generate(messages)
    call = parse_tool_call(text)  # Invalid or unknown calls never reach the backend.
    additions = [{"role": "assistant", "content": text}]
    if call is None:
        return additions
    try:
        results = normalize_results(search(call["arguments"]["query"]))
        payload = {"name": "web_search", "status": "ok" if results else "empty", "results": results}
    except Exception:
        # Do not leak provider errors, credentials, URLs with secrets, or tracebacks.
        payload = {"name": "web_search", "status": "error", "results": []}
    payload["calls_remaining"] = 0
    additions.append({"role": "tool", "content": safe_json(payload)})
    final = generate(messages + additions)
    if parse_tool_call(final) is not None:
        raise ValueError("Search call budget exhausted; the model did not produce a final answer")
    additions.append({"role": "assistant", "content": final})
    return additions
