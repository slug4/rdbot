import time
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from .core import validate_url

USER_AGENT = 'RxDxResearchBot/0.1'
MAX_BYTES = 2_000_000


def fetch(url, session, authorize=None):
    """Follow only allowlisted redirects, with TLS verification and bounded reads."""
    for _ in range(5):
        validate_url(url)
        if authorize and not authorize(url):
            raise ValueError("Redirect target is not approved by origin and robots rules.")
        with session.get(url, headers={'User-Agent': USER_AGENT}, timeout=25,
                         allow_redirects=False, stream=True) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.headers.get('Location', ''))
                continue
            response.raise_for_status()
            if not any(kind in response.headers.get('Content-Type', '').lower() for kind in ('text/', 'application/xhtml+xml')):
                raise ValueError('Source did not return text or HTML.')
            parts, total = [], 0
            for part in response.iter_content(8192):
                total += len(part)
                if total > MAX_BYTES:
                    raise ValueError('Source exceeds 2 MB limit.')
                parts.append(part)
            return b''.join(parts).decode(response.encoding or 'utf-8', errors='replace'), url
    raise ValueError('Too many redirects.')


def crawl(urls, session=None):
    """Fetch manually selected pages only; fail closed when robots is unavailable."""
    if not urls or len(urls) > 30:
        raise ValueError('Select 1–30 approved source URLs.')
    session = session or requests.Session()
    rules, pages = {}, []
    for url in dict.fromkeys(urls):
        validate_url(url)
        origin = f'https://{urlsplit(url).netloc}'
        if origin not in rules:
            text, _ = fetch(origin + '/robots.txt', session)
            if '<html' in text.lower() or '<!doctype' in text.lower():
                raise ValueError('robots.txt returned HTML instead of robots rules.')
            robot = RobotFileParser()
            robot.parse(text.splitlines())
            rules[origin] = robot
        robot = rules[origin]
        if not robot.can_fetch(USER_AGENT, url):
            raise ValueError(f'robots.txt does not permit fetching {url}')
        delay = max(1, robot.crawl_delay(USER_AGENT) or 0)
        if delay > 30:
            raise ValueError('Source requires a longer crawl delay; use a separately scheduled indexer.')
        time.sleep(delay)
        html, final_url = fetch(url, session, authorize=lambda target: f'https://{urlsplit(target).netloc}' == origin and robot.can_fetch(USER_AGENT, target))
        final_origin = f'https://{urlsplit(final_url).netloc}'
        if final_origin != origin:
            raise ValueError('Cross-origin redirect requires explicitly configuring the final source URL.')
        if not robot.can_fetch(USER_AGENT, final_url):
            raise ValueError('Redirect target is disallowed by robots.txt.')
        soup = BeautifulSoup(html, 'html.parser')
        for node in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
            node.decompose()
        text = (soup.find('main') or soup).get_text(' ', strip=True)
        if len(text) < 80:
            raise ValueError(f'No useful public content extracted from {final_url}')
        pages.append((final_url, text))
    return pages
