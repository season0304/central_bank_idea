"""Import BIS yearly speech archives, retaining source and ambiguous records."""
import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[1] / 'data' / 'central_bank_speeches'
PATTERNS = {
    'euro_area': r'European Central Bank|\bECB\b',
    'germany': r'Deutsche Bundesbank|German Bundesbank|\bBundesbank\b',
    'france': r'Bank of France|Banque de France',
    'italy': r'Bank of Italy|Banca d.Italia',
    'united_kingdom': r'Bank of England',
    'japan': r'Bank of Japan',
    'canada': r'Bank of Canada',
    'south_korea': r'Bank of Korea',
    'china': r"People.s Bank of China|\bPBoC\b|\bPBOC\b",
    'australia': r'Reserve Bank of Australia',
    'india': r'Reserve Bank of India',
}

def main(paths):
    counts = {k: 0 for k in PATTERNS}
    ambiguous = []
    for archive in paths:
        with zipfile.ZipFile(archive) as z:
            name = next(x for x in z.namelist() if x.endswith('.csv'))
            rows = csv.DictReader(io.TextIOWrapper(z.open(name), encoding='utf-8-sig', newline=''))
            for row in rows:
                # The speaker's affiliation usually precedes the event description.
                affiliation = re.split(r',\s+(?:at|delivered at|during|on behalf of)\b', row['description'], maxsplit=1, flags=re.I)[0]
                hits = [key for key, pattern in PATTERNS.items() if re.search(pattern, affiliation, re.I)]
                if len(hits) > 1:
                    ambiguous.append({**row, 'candidate_institutions': ', '.join(hits)})
                    continue
                if not hits or not row['text'].strip():
                    continue
                country = hits[0]
                date = row['date'][:10]
                if not re.fullmatch(r'20(?:24|25|26)-\d\d-\d\d', date):
                    continue
                uid = hashlib.sha256(row['url'].encode()).hexdigest()[:12]
                name = re.sub(r'[^a-z0-9]+', '-', row['author'].lower()).strip('-')[:50] or 'unknown'
                target = ROOT / country / date[:4] / f"{date}_{name}_bis_{uid}.md"
                text = (f"# {row['title']}\n\n- Date: {date}\n- Speaker: {row['author']}\n"
                        f"- Affiliation as published: {affiliation}\n- Source URL: {row['url']}\n"
                        f"- Source dataset: BIS central bankers' speeches ({Path(archive).name})\n\n"
                        + row['text'].strip() + '\n')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding='utf-8')
                counts[country] += 1
    print('Imported by institution:', counts)
    print('Ambiguous affiliation records skipped:', len(ambiguous))
    if ambiguous:
        path = ROOT / 'bis_ambiguous_affiliations.csv'
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'author', 'title', 'description', 'url', 'candidate_institutions'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(ambiguous)
        print('Review:', path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python3 scripts/import_bis_speeches.py <yearly-bis-archive.zip> ...')
    main(sys.argv[1:])
