"""
Royce News Scanner
------------------
Scans the internet for breaking news and current events.
Uses NewsAPI when available, falls back to RSS/web scraping.
"""

import os
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# RSS feeds for fallback
RSS_FEEDS = {
    "top_news": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/reuters/topNews",
    ],
    "tech": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://www.theverge.com/rss/index.xml",
    ],
    "world": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
    ],
    "science": [
        "https://www.sciencedaily.com/rss/all.xml",
    ],
}


class RoyceNews:
    """Scans the internet for breaking news and current events."""

    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.news_api_url = "https://newsapi.org/v2"
        self.cached_headlines = []
        self.last_fetch = None
        self.cache_duration = timedelta(minutes=15)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        logger.info("Royce News Scanner initialized")

    # ─── Main Headlines ────────────────────────────────────────────

    def get_breaking_news(self, category: Optional[str] = None, count: int = 10) -> Dict:
        """Get the latest breaking news headlines.

        Args:
            category: Optional category filter (technology, sports, science, health, business, entertainment)
            count: Number of headlines to return

        Returns:
            Dict with headlines and metadata
        """
        # Check cache
        if self._cache_valid():
            headlines = self.cached_headlines
            if category:
                headlines = [h for h in headlines if category.lower() in h.get("category", "").lower()]
            return self._format_results(headlines[:count])

        # Try NewsAPI first
        if self.news_api_key:
            result = self._fetch_from_newsapi(category, count)
            if result["status"] == "success":
                return result

        # Fallback to RSS
        return self._fetch_from_rss(category, count)

    def search_news(self, query: str, count: int = 10) -> Dict:
        """Search news for a specific topic.

        Args:
            query: Search term
            count: Number of results

        Returns:
            Dict with matching articles
        """
        if self.news_api_key:
            return self._search_newsapi(query, count)
        return self._search_web(query, count)

    # ─── NewsAPI Methods ───────────────────────────────────────────

    def _fetch_from_newsapi(self, category: Optional[str] = None, count: int = 10) -> Dict:
        """Fetch headlines from NewsAPI."""
        try:
            params = {
                "apiKey": self.news_api_key,
                "country": "us",
                "pageSize": count,
            }
            if category:
                params["category"] = category

            response = requests.get(
                f"{self.news_api_url}/top-headlines",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "ok":
                articles = []
                for article in data.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "published": article.get("publishedAt", ""),
                        "category": category or "general",
                    })
                self.cached_headlines = articles
                self.last_fetch = datetime.now()
                return self._format_results(articles[:count])

        except Exception as e:
            logger.error(f"NewsAPI error: {e}")

        return {"status": "error", "message": "Couldn't fetch from NewsAPI"}

    def _search_newsapi(self, query: str, count: int = 10) -> Dict:
        """Search NewsAPI for a topic."""
        try:
            params = {
                "apiKey": self.news_api_key,
                "q": query,
                "pageSize": count,
                "sortBy": "publishedAt",
                "language": "en",
            }

            response = requests.get(
                f"{self.news_api_url}/everything",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "ok":
                articles = []
                for article in data.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "published": article.get("publishedAt", ""),
                    })
                return self._format_results(articles[:count])

        except Exception as e:
            logger.error(f"NewsAPI search error: {e}")

        return self._search_web(query, count)

    # ─── RSS Fallback ──────────────────────────────────────────────

    def _fetch_from_rss(self, category: Optional[str] = None, count: int = 10) -> Dict:
        """Fallback: fetch from RSS feeds."""
        feed_key = category if category in RSS_FEEDS else "top_news"
        feeds = RSS_FEEDS.get(feed_key, RSS_FEEDS["top_news"])

        articles = []
        for feed_url in feeds:
            try:
                response = self.session.get(feed_url, timeout=8)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "xml")

                for item in soup.find_all("item")[:count]:
                    title = item.find("title")
                    desc = item.find("description")
                    link = item.find("link")
                    pub_date = item.find("pubDate")

                    articles.append({
                        "title": title.get_text().strip() if title else "",
                        "source": feed_url.split("/")[2],
                        "description": desc.get_text().strip()[:200] if desc else "",
                        "url": link.get_text().strip() if link else "",
                        "published": pub_date.get_text().strip() if pub_date else "",
                        "category": feed_key,
                    })

                    if len(articles) >= count:
                        break
            except Exception as e:
                logger.warning(f"RSS feed error ({feed_url}): {e}")
                continue

            if len(articles) >= count:
                break

        if articles:
            self.cached_headlines = articles
            self.last_fetch = datetime.now()
            return self._format_results(articles[:count])

        return {"status": "error", "message": "Couldn't pull any news right now. Check the connection."}

    # ─── Web Search Fallback ───────────────────────────────────────

    def _search_web(self, query: str, count: int = 10) -> Dict:
        """Fallback: search for news via Google News."""
        try:
            url = f"https://news.google.com/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            articles = []
            for article_tag in soup.find_all("article")[:count]:
                title_tag = article_tag.find("a", class_=True)
                if title_tag:
                    articles.append({
                        "title": title_tag.get_text().strip(),
                        "source": "Google News",
                        "description": "",
                        "url": "",
                        "published": "",
                    })

            if articles:
                return self._format_results(articles)
        except Exception as e:
            logger.error(f"Google News search error: {e}")

        return {"status": "error", "message": f"Couldn't find news about '{query}' right now."}

    # ─── Formatting ────────────────────────────────────────────────

    def _format_results(self, articles: List[Dict]) -> Dict:
        """Format news results into a clean response."""
        if not articles:
            return {"status": "empty", "message": "No news articles found."}

        formatted = []
        for i, article in enumerate(articles, 1):
            formatted.append({
                "number": i,
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "published": article.get("published", ""),
            })

        # Build a readable summary
        summary_lines = []
        for a in formatted:
            line = f"{a['number']}. {a['title']}"
            if a["source"]:
                line += f" ({a['source']})"
            summary_lines.append(line)

        return {
            "status": "success",
            "count": len(formatted),
            "articles": formatted,
            "summary": "\n".join(summary_lines),
            "fetched_at": datetime.now().isoformat(),
        }

    def _cache_valid(self) -> bool:
        """Check if cached headlines are still fresh."""
        if not self.last_fetch or not self.cached_headlines:
            return False
        return datetime.now() - self.last_fetch < self.cache_duration

    # ─── Command Parsing ───────────────────────────────────────────

    def parse_news_command(self, text: str) -> Optional[Dict]:
        """Detect news-related requests from natural language."""
        text_lower = text.lower().strip()

        # Breaking news requests
        breaking_patterns = [
            r"(?:what's|whats) (?:the |)(?:latest |breaking |)news",
            r"(?:any |)breaking news",
            r"what's (?:happening|going on) (?:in the world|today|right now)",
            r"(?:give me|show me|get me) (?:the |)(?:latest |)news",
            r"news (?:update|briefing|headlines)",
            r"catch me up (?:on )?(?:the news|what's happening)",
            r"what did i miss",
        ]
        for pattern in breaking_patterns:
            if re.search(pattern, text_lower):
                return {"action": "headlines"}

        # Category-specific news
        categories = {
            "tech": ["tech news", "technology news", "tech headlines"],
            "science": ["science news", "science headlines"],
            "sports": ["sports news", "sports headlines", "sports scores"],
            "business": ["business news", "market news", "financial news"],
            "health": ["health news", "medical news"],
            "entertainment": ["entertainment news", "celebrity news", "movie news"],
        }
        for category, triggers in categories.items():
            if any(trigger in text_lower for trigger in triggers):
                return {"action": "headlines", "category": category}

        # News search
        search_patterns = [
            r"news (?:about|on|regarding)\s+(.+)",
            r"(?:any|latest) news (?:about|on)\s+(.+)",
            r"what's (?:the news|happening) with\s+(.+)",
        ]
        for pattern in search_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return {"action": "search", "query": match.group(1).strip()}

        return None
