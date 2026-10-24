
"""Public news discovery helpers: GDELT by default, optional NewsAPI."""

from __future__ import annotations

import os
import time
from urllib.parse import urlparse

import requests

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

HEADERS = {
    "User-Agent": "NewsGuardAI/1.0 (news research and source discovery)"
}


def get_json_with_retry(
    url: str,
    params: dict,
    timeout: int = 25,
    max_retries: int = 3,
) -> dict:
    """Request JSON with backoff for rate limits and temporary server errors."""

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=timeout,
            )

            if response.status_code == 429:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "GDELT is rate-limiting requests (HTTP 429). "
                        "Wait a few minutes before searching again. "
                        "You do not need an API key to use GDELT."
                    )

                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 3 * (2 ** attempt)
                except ValueError:
                    delay = 3 * (2 ** attempt)

                time.sleep(min(max(delay, 1), 30))
                continue

            if response.status_code in (500, 502, 503, 504):
                if attempt == max_retries - 1:
                    response.raise_for_status()

                time.sleep(2 ** (attempt + 1))
                continue

            response.raise_for_status()
            return response.json()

        except requests.Timeout:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    "The news provider timed out. Please try again later."
                )
            time.sleep(2 ** (attempt + 1))

        except requests.RequestException as exc:
            raise RuntimeError(f"News provider request failed: {exc}") from exc

    raise RuntimeError("News search failed after retries.")


def search_news(
    query: str,
    limit: int = 10,
    provider: str = "GDELT",
) -> list[dict]:
    """Search public news metadata using GDELT or optional NewsAPI."""

    query = (query or "").strip()

    if len(query) < 3:
        raise ValueError("Enter at least 3 characters to search for news.")

    limit = max(1, min(int(limit), 25))
    provider = (provider or "GDELT").strip().upper()

    if provider == "NEWSAPI":
        key = os.getenv("NEWSAPI_KEY", "").strip()

        if not key:
            raise RuntimeError(
                "NewsAPI requires a key. Select GDELT in the app, "
                "or configure NEWSAPI_KEY if you have a NewsAPI key."
            )

        payload = get_json_with_retry(
            NEWSAPI_ENDPOINT,
            {
                "q": query,
                "pageSize": limit,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": key,
            },
            timeout=25,
        )

        if payload.get("status") != "ok":
            raise RuntimeError(
                payload.get("message", "NewsAPI request failed.")
            )

        results = []

        for article in payload.get("articles", []):
            url = article.get("url") or ""
            if not url:
                continue

            results.append({
                "title": article.get("title") or "Untitled",
                "source": (
                    (article.get("source") or {}).get("name")
                    or urlparse(url).netloc
                ),
                "published_at": article.get("publishedAt") or "",
                "url": url,
                "description": article.get("description") or "",
                "provider": "NewsAPI",
            })

        return results

    if provider != "GDELT":
        raise ValueError("Provider must be GDELT or NewsAPI.")

    payload = get_json_with_retry(
        GDELT_ENDPOINT,
        {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": limit,
            "sort": "HybridRel",
            "sourcelang": "english",
        },
        timeout=30,
    )

    results = []

    for article in payload.get("articles", []) or []:
        url = article.get("url") or ""
        if not url:
            continue

        results.append({
            "title": article.get("title") or "Untitled",
            "source": article.get("domain") or urlparse(url).netloc,
            "published_at": article.get("seendate") or "",
            "url": url,
            "description": article.get("socialimage") or "",
            "provider": "GDELT",
        })

    return results
