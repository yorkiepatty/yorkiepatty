"""
Royce Deep Research Engine
--------------------------
Performs thorough web research on any subject or person.
Uses Perplexity API for AI-powered search, falls back to web scraping.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RoyceResearch:
    """Deep web research engine — digs into any topic or person."""

    def __init__(self):
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        self.perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        logger.info("Royce Research Engine initialized")

    # ─── Perplexity AI Search ──────────────────────────────────────

    def deep_search(self, query: str, detail_level: str = "thorough") -> Dict:
        """Perform a deep research dive using Perplexity AI.

        Args:
            query: What to research
            detail_level: "quick", "thorough", or "exhaustive"

        Returns:
            Dict with research results, sources, and summary
        """
        if self.perplexity_key:
            return self._perplexity_search(query, detail_level)
        else:
            return self._web_scrape_search(query)

    def _perplexity_search(self, query: str, detail_level: str = "thorough") -> Dict:
        """Use Perplexity AI for deep research."""
        detail_prompts = {
            "quick": "Give a concise summary of this topic. Be direct and factual.",
            "thorough": "Do a thorough research dive on this topic. Cover key facts, background, current status, and notable details. Be comprehensive but readable.",
            "exhaustive": "Do an exhaustive deep dive on this topic. Leave no stone unturned. Cover history, current state, key players, controversies, statistics, and predictions. This should be the most thorough briefing possible."
        }

        system_prompt = f"""You are a research assistant for Royce, a powerful AI.
{detail_prompts.get(detail_level, detail_prompts['thorough'])}
Always cite your sources. Present information in a clear, organized way.
If this is about a person, include key biographical info, notable achievements, recent news, and public perception."""

        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.perplexity_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "max_tokens": 2000 if detail_level == "quick" else 4000,
        }

        try:
            response = requests.post(
                self.perplexity_url, headers=headers, json=data, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            citations = result.get("citations", [])

            return {
                "status": "success",
                "query": query,
                "detail_level": detail_level,
                "content": content,
                "sources": citations,
                "timestamp": datetime.now().isoformat(),
            }
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "Research request timed out. Try again."}
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API error: {e}")
            # Fall back to web scraping
            return self._web_scrape_search(query)

    # ─── Web Scrape Fallback ───────────────────────────────────────

    def _web_scrape_search(self, query: str) -> Dict:
        """Fallback: scrape Google search results and extract content."""
        try:
            search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract search result links
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith("/url?q="):
                    url = href.split("/url?q=")[1].split("&")[0]
                    if not any(skip in url for skip in ["google.com", "youtube.com", "webcache"]):
                        links.append(url)
                        if len(links) >= 5:
                            break

            # Extract content from top links
            results = []
            for url in links[:3]:
                try:
                    page = self.session.get(url, timeout=8)
                    page_soup = BeautifulSoup(page.text, "html.parser")

                    # Remove script and style elements
                    for tag in page_soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()

                    title = page_soup.title.string if page_soup.title else "No title"
                    paragraphs = page_soup.find_all("p")
                    text = " ".join(p.get_text().strip() for p in paragraphs[:10])
                    text = re.sub(r'\s+', ' ', text)[:1000]

                    if text:
                        results.append({
                            "title": title.strip(),
                            "url": url,
                            "excerpt": text
                        })
                except Exception:
                    continue

            if results:
                combined = "\n\n".join(
                    f"**{r['title']}**\n{r['excerpt']}" for r in results
                )
                return {
                    "status": "success",
                    "query": query,
                    "detail_level": "web_scrape",
                    "content": combined,
                    "sources": [r["url"] for r in results],
                    "timestamp": datetime.now().isoformat(),
                }

            return {"status": "error", "message": "Couldn't find useful results. Try rephrasing."}

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"status": "error", "message": f"Search failed: {str(e)}"}

    # ─── Person Research ───────────────────────────────────────────

    def research_person(self, name: str) -> Dict:
        """Specialized deep dive on a person."""
        query = f"Who is {name}? Biography, career, recent news, notable achievements, public life"
        return self.deep_search(query, detail_level="thorough")

    # ─── Topic Deep Dive ───────────────────────────────────────────

    def research_topic(self, topic: str) -> Dict:
        """Specialized deep dive on any subject."""
        query = f"Comprehensive overview of {topic}: history, current state, key facts, recent developments"
        return self.deep_search(query, detail_level="thorough")

    # ─── Command Parsing ───────────────────────────────────────────

    def parse_research_command(self, text: str) -> Optional[Dict]:
        """Detect if the user wants research done.

        Returns action dict or None.
        """
        text_lower = text.lower().strip()

        # Person lookups
        person_patterns = [
            r"who is\s+(.+?)[\?]?$",
            r"look up\s+(.+)",
            r"tell me about\s+(.+)",
            r"research\s+(.+)",
            r"what do you know about\s+(.+?)[\?]?$",
            r"find out about\s+(.+)",
            r"deep dive (?:on|into)\s+(.+)",
            r"dig into\s+(.+)",
        ]

        for pattern in person_patterns:
            match = re.match(pattern, text_lower)
            if match:
                subject = match.group(1).strip().rstrip("?.")
                return {"action": "research", "subject": subject}

        return None
