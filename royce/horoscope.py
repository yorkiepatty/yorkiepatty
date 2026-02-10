"""
Royce Horoscope Reader
----------------------
Fetches daily horoscopes for any zodiac sign.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ZODIAC_SIGNS = {
    "aries": {"dates": "Mar 21 - Apr 19", "element": "Fire"},
    "taurus": {"dates": "Apr 20 - May 20", "element": "Earth"},
    "gemini": {"dates": "May 21 - Jun 20", "element": "Air"},
    "cancer": {"dates": "Jun 21 - Jul 22", "element": "Water"},
    "leo": {"dates": "Jul 23 - Aug 22", "element": "Fire"},
    "virgo": {"dates": "Aug 23 - Sep 22", "element": "Earth"},
    "libra": {"dates": "Sep 23 - Oct 22", "element": "Air"},
    "scorpio": {"dates": "Oct 23 - Nov 21", "element": "Water"},
    "sagittarius": {"dates": "Nov 22 - Dec 21", "element": "Fire"},
    "capricorn": {"dates": "Dec 22 - Jan 19", "element": "Earth"},
    "aquarius": {"dates": "Jan 20 - Feb 18", "element": "Air"},
    "pisces": {"dates": "Feb 19 - Mar 20", "element": "Water"},
}

# Map birthday month/day to zodiac sign
ZODIAC_DATE_RANGES = [
    (1, 20, "capricorn"), (2, 18, "aquarius"), (3, 20, "pisces"),
    (4, 19, "aries"), (5, 20, "taurus"), (6, 20, "gemini"),
    (7, 22, "cancer"), (8, 22, "leo"), (9, 22, "virgo"),
    (10, 22, "libra"), (11, 21, "scorpio"), (12, 21, "sagittarius"),
    (12, 31, "capricorn"),
]


class RoyceHoroscope:
    """Fetches and delivers daily horoscope readings."""

    def __init__(self):
        self.cache = {}  # Simple daily cache
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        logger.info("Royce Horoscope Reader initialized")

    def get_horoscope(self, sign: str) -> Dict:
        """Get today's horoscope for a zodiac sign.

        Args:
            sign: Zodiac sign name (e.g., "aries", "leo")

        Returns:
            Dict with horoscope text and metadata
        """
        sign = sign.lower().strip()
        if sign not in ZODIAC_SIGNS:
            return {"status": "error", "message": f"'{sign}' isn't a zodiac sign I know. Try one of: {', '.join(ZODIAC_SIGNS.keys())}"}

        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{sign}_{today}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self._fetch_horoscope(sign)
        if result["status"] == "success":
            self.cache[cache_key] = result
        return result

    def _fetch_horoscope(self, sign: str) -> Dict:
        """Fetch horoscope from web sources."""
        # Try Horoscope.com
        try:
            today = datetime.now()
            url = f"https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign={list(ZODIAC_SIGNS.keys()).index(sign) + 1}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            # Look for the horoscope text container
            content_div = soup.find("div", class_="main-horoscope")
            if content_div:
                paragraphs = content_div.find_all("p")
                text = " ".join(p.get_text().strip() for p in paragraphs)
                text = re.sub(r'\s+', ' ', text).strip()
                # Remove date prefix if present
                text = re.sub(r'^[A-Za-z]+\s+\d+,\s+\d+\s*[-–]\s*', '', text)

                if text:
                    info = ZODIAC_SIGNS[sign]
                    return {
                        "status": "success",
                        "sign": sign.capitalize(),
                        "dates": info["dates"],
                        "element": info["element"],
                        "horoscope": text,
                        "date": today.strftime("%B %d, %Y"),
                    }
        except Exception as e:
            logger.warning(f"Horoscope.com fetch failed: {e}")

        # Fallback: try astrology.com
        try:
            url = f"https://www.astrology.com/horoscope/daily/{sign}.html"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", {"id": "content"}) or soup.find("main")
            if content:
                paragraphs = content.find_all("p")
                text = " ".join(p.get_text().strip() for p in paragraphs[:3])
                text = re.sub(r'\s+', ' ', text).strip()

                if text:
                    info = ZODIAC_SIGNS[sign]
                    return {
                        "status": "success",
                        "sign": sign.capitalize(),
                        "dates": info["dates"],
                        "element": info["element"],
                        "horoscope": text,
                        "date": datetime.now().strftime("%B %d, %Y"),
                    }
        except Exception as e:
            logger.warning(f"Astrology.com fetch failed: {e}")

        return {
            "status": "error",
            "message": f"Couldn't pull up the horoscope for {sign.capitalize()} right now. Try again in a bit."
        }

    @staticmethod
    def get_sign_from_birthday(month: int, day: int) -> Optional[str]:
        """Determine zodiac sign from birthday month and day."""
        for end_month, end_day, sign in ZODIAC_DATE_RANGES:
            if month < end_month or (month == end_month and day <= end_day):
                return sign
        return "capricorn"

    def parse_horoscope_command(self, text: str) -> Optional[Dict]:
        """Detect horoscope requests from natural language."""
        text_lower = text.lower().strip()

        # Direct sign requests
        for sign in ZODIAC_SIGNS:
            patterns = [
                rf"\b{sign}\b.*horoscope",
                rf"horoscope.*\b{sign}\b",
                rf"read.*\b{sign}\b",
                rf"what's.*\b{sign}\b.*say",
                rf"\b{sign}\b.*today",
            ]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return {"action": "horoscope", "sign": sign}

        # General horoscope requests (user's sign needed from memory)
        general_patterns = [
            r"(?:my|read my|what's my) horoscope",
            r"horoscope (?:for )?today",
            r"daily horoscope",
            r"what do the stars say",
        ]
        for pattern in general_patterns:
            if re.search(pattern, text_lower):
                return {"action": "horoscope", "sign": None}  # Will need to look up from memory

        return None
