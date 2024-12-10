from abc import ABC, abstractmethod
import re
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional

class HTMLParser(ABC):
    """Base class for parsing PornHub HTML snapshots"""

    @abstractmethod
    def parse(self, html: str) -> List[Dict]:
        """Parse HTML and return list of video data"""
        pass

    def can_parse(self, html: str) -> bool:
        """Check if this parser can handle the given HTML"""
        try:
            results = self.parse(html)
            if len(results) == 0:
                print(f"{self.__class__.__name__}: No results found")
                return False
            return True
        except Exception as e:
            print(f"{self.__class__.__name__} parsing failed: {str(e)}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace"""
        return re.sub(r'\s+', ' ', text.strip())

####### Subclass

class Parser2008(HTMLParser):
    """Parser for 2008 HTML format"""
    def parse(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, 'html.parser')
        videos = []

        # 2008 format used div.video_box with nested small class for titles
        for video in soup.select('.video_box'):
            title_elem = video.select_one('.small')
            url_elem = video.select_one('a')
            views_elem = video.select_one('.box-right')
            duration_elem = video.select_one('.box-left')

            if title_elem and url_elem:
                video_data = {
                    'title': self._clean_text(title_elem.get_text()),
                    'url': url_elem.get('href', ''),
                }

                # Extract views if available
                if views_elem:
                    views_text = views_elem.get_text()
                    if 'views' in views_text:
                        video_data['views'] = views_text.split()[0].strip()

                # Extract duration if available
                if duration_elem:
                    video_data['duration'] = duration_elem.get_text().strip()

                videos.append(video_data)

        return videos

class Parser2010(HTMLParser):
    """Parser for 2010 HTML format"""
    def parse(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, 'html.parser')
        videos = []

        # 2010 format used div.wrap with title/variable classes
        for video in soup.select('.wrap'):
            title_elem = video.select_one('.title a')
            views_elem = video.select_one('.views')
            duration_elem = video.select_one('.duration')

            if title_elem:
                video_data = {
                    'title': self._clean_text(title_elem.get_text()),
                    'url': title_elem.get('href', '')
                }

                if views_elem:
                    views_text = views_elem.get_text()
                    if 'views' in views_text:
                        video_data['views'] = views_text.split()[0].strip()

                if duration_elem:
                    video_data['duration'] = duration_elem.get_text().strip()

                videos.append(video_data)

        return videos

class Parser2012(HTMLParser):
    """Parser for 2012 HTML format"""
    def parse(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, 'html.parser')
        videos = []

        # 2012 format used wrap class with thumbnail-info-wrapper
        for video in soup.select('.wrap'):
            title_elem = video.select_one('.thumbnail-info-wrapper .title a')
            views_text = video.select_one('.views')
            duration = video.select_one('.duration')

            if title_elem:
                video_data = {
                    'title': self._clean_text(title_elem.get_text()),
                    'url': title_elem.get('href', '')
                }

                if views_text:
                    views = views_text.get_text().split()[0].strip()
                    video_data['views'] = views.replace(',', '')

                if duration:
                    video_data['duration'] = duration.get_text().strip()

                videos.append(video_data)

        return videos

class Parser2022(HTMLParser):
    """Parser for 2022 HTML format"""

    def parse(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, 'html.parser')
        videos = []

        for video in soup.select('.videoBox'):
            title_elem = video.select_one('.title a')
            if title_elem:
                videos.append({
                    'title': self._clean_text(title_elem.get_text()),
                    'url': title_elem.get('href', '')
                })

        return videos

######## Fact

class ParserFactory:
    """Factory for creating appropriate parser based on content"""

    @staticmethod
    def get_parser(html: str, date: datetime) -> Optional[HTMLParser]:
        """Try each parser and return first one that works"""
        parsers = [Parser2008(), Parser2010(), Parser2012(), Parser2022()]

        for parser in parsers:
            if parser.can_parse(html):
                return parser

        print(f"No working parser found for date: {date}")
        return None

###### Func

def parse_snapshot(html: str, date: datetime) -> List[Dict]:
    """Parse a snapshot from a specific date"""
    parser = ParserFactory.get_parser(html, date)
    if parser:
        return parser.parse(html)
    return []
