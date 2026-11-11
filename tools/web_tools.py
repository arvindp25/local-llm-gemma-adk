import logging
import re
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# Try importing ddgs first, fallback to duckduckgo_search
try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


def web_search(query: str, max_results: int = 5) -> str:
    """Searches the live internet via DuckDuckGo to obtain up-to-date web pages, facts, and documentation.
    
    Args:
        query: Search keywords or question to look up online.
        max_results: Maximum search results to return (default is 5).
    """
    if not HAS_DDGS:
        return "Error: Neither `ddgs` nor `duckduckgo_search` library is installed."
    
    try:
        clean_query = query.strip()
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(clean_query, max_results=max_results))
        
        if not raw_results:
            return f"No internet search results found for: '{query}'."

        formatted_results = []
        for i, item in enumerate(raw_results, 1):
            title = item.get("title", "No Title")
            snippet = item.get("body", "")
            href = item.get("href", "")
            formatted_results.append(
                f"[{i}] {title}\nURL: {href}\nSnippet: {snippet}"
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Internet search error for query '{query}': {e}")
        return f"Error executing internet search: {str(e)}"


def fetch_web_page(url: str, max_chars: int = 3500) -> str:
    """Fetches and extracts clean, readable text from a web page on the internet.
    
    Args:
        url: Absolute HTTP or HTTPS URL to load.
        max_chars: Maximum characters of text content to return (default 3500).
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: Invalid URL '{url}'. Must start with http:// or https://."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html_content = resp.text

        if HAS_BS4:
            soup = BeautifulSoup(html_content, "html.parser")
            # Remove scripts, styles, forms, and tracking tags
            for unwanted in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
                unwanted.decompose()
            text = soup.get_text(separator="\n")
        else:
            # Fallback regex strip
            text = re.sub(r"<[^>]+>", " ", html_content)

        # Collapse whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + f"\n\n[Content truncated at {max_chars} characters...]"

        return clean_text or "Notice: Web page content was empty."
    except Exception as e:
        logger.error(f"Error fetching web page '{url}': {e}")
        return f"Error loading URL '{url}': {str(e)}"
