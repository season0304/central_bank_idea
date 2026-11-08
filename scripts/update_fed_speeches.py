"""Download Federal Reserve Board speeches from the official yearly archives.

Run periodically: python3 scripts/update_fed_speeches.py
The Board archive does not cover every Reserve Bank president's speech.
"""
import argparse
import concurrent.futures
import datetime as dt
import html.parser
import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = 'https://www.federalreserve.gov'
ROOT = Path(__file__).resolve().parents[1] / 'data' / 'fed_speeches'
LINK = re.compile(r'/newsevents/speech/[a-z]+(\d{8})[a-z]\.htm', re.I)


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicResearchSpeechCollector/1.0 (public archive)'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode('utf-8-sig', errors='replace')


class Article(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.level = 0
        self.part = None
        self.tag = None
        self.content = []
        self.result = {'title': '', 'speaker': '', 'paragraphs': []}
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div' and attrs.get('id') == 'article':
            self.level = 1
            return
        if not self.level:
            return
        if tag == 'div':
            self.level += 1
        if tag == 'sup':
            self.skip += 1
        if self.part is None and (tag == 'h3' and 'title' in attrs.get('class', '') or tag == 'p'):
            self.part = 'title' if tag == 'h3' else ('speaker' if 'speaker' in attrs.get('class', '').split() else 'paragraphs')
            self.tag = tag
            self.content = []

    def handle_endtag(self, tag):
        if not self.level:
            return
        if tag == 'sup' and self.skip:
            self.skip -= 1
        if self.part and tag == self.tag:
            value = ' '.join(''.join(self.content).split())
            if value:
                if self.part == 'paragraphs':
                    self.result['paragraphs'].append(value)
                else:
                    self.result[self.part] = value
            self.part = self.tag = None
        if tag == 'div':
            self.level -= 1

    def handle_data(self, data):
        if self.part and not self.skip:
            self.content.append(data)


def save(url):
    match = LINK.search(url)
    date = dt.datetime.strptime(match.group(1), '%Y%m%d').date()
    target = ROOT / str(date.year) / (Path(url).stem + '.md')
    if target.exists():
        return 'existing', str(target)
    parser = Article()
    parser.feed(fetch(url))
    article = parser.result
    if not article['title'] or not article['speaker'] or not article['paragraphs']:
        raise ValueError('Could not extract title, speaker and text: ' + url)
    # Retain source text including footnotes; omit the date and event-location header.
    paragraphs = [p for p in article['paragraphs'] if not re.fullmatch(r'[A-Za-z]+ \d{1,2}, \d{4}', p) and not p.startswith('At the ')]
    text = f"# {article['title']}\n\n- Date: {date}\n- Speaker: {article['speaker']}\n- Source: {url}\n- Collection: Federal Reserve Board speeches\n\n" + '\n\n'.join(paragraphs) + '\n'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    return 'downloaded', str(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start-year', type=int, default=2024)
    ap.add_argument('--end-year', type=int, default=dt.date.today().year)
    args = ap.parse_args()
    urls = set()
    for year in range(args.start_year, args.end_year + 1):
        archive = f'{BASE}/newsevents/{year}-speeches.htm'
        page = fetch(archive)
        urls.update(BASE + m.group(0) for m in LINK.finditer(page) if int(m.group(1)[:4]) == year)
        print(f'{year}: {sum(1 for u in urls if str(year) in u)} speech links', flush=True)
    errors = []
    counts = {'downloaded': 0, 'existing': 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(save, url): url for url in sorted(urls)}
        for future in concurrent.futures.as_completed(futures):
            try:
                status, _ = future.result()
                counts[status] += 1
            except Exception as exc:
                errors.append({'url': futures[future], 'error': str(exc)})
    print(json.dumps({'counts': counts, 'errors': errors}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
