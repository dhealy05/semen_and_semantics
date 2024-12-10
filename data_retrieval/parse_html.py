from pathlib import Path
from typing import Iterator, Tuple
from datetime import datetime
import json
from bs4 import BeautifulSoup
#
from parser import parse_snapshot

def iterate_snapshots(base_dir: str, website: str) -> Iterator[Tuple[datetime, str, Path]]:
    """
    Iterates over all snapshot files for a given website in chronological order.
    Returns tuples of (datetime, html_content, file_path)
    """
    site_dir = website.replace('http://', '').replace('https://', '').rstrip('/')
    site_dir = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in site_dir)
    website_path = Path(base_dir) / site_dir

    if not website_path.exists():
        raise FileNotFoundError(f"No snapshots found for {website} in {base_dir}")

    # Collect all files and their dates first
    snapshot_files = []
    for html_file in website_path.glob("*/*/*.html"):
        try:
            date_str = html_file.stem
            snapshot_date = datetime.strptime(date_str, "%Y%m%d")
            snapshot_files.append((snapshot_date, html_file))
        except ValueError as e:
            print(f"Error processing {html_file}: {e}")
            continue

    # Sort by date
    snapshot_files.sort(key=lambda x: x[0])

    # Yield sorted snapshots
    for snapshot_date, html_file in snapshot_files:
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            yield snapshot_date, html_content, html_file
        except IOError as e:
            print(f"Error reading {html_file}: {e}")
            continue

def save_json(data: list, filepath: Path) -> None:
    """Save parsed data as JSON next to source HTML"""
    json_path = filepath.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    base_dir = "../snapshots"
    website = "pornhub.com"

    try:
        for snapshot_date, html_content, filepath in iterate_snapshots(base_dir, website):
            print(f"Processing {filepath}")

            # Parse HTML using our parser framework
            video_data = parse_snapshot(html_content, snapshot_date)

            if video_data:
                save_json(video_data, filepath)
                print(f"Saved {len(video_data)} videos to {filepath.with_suffix('.json')}")
            else:
                print(f"No videos found in {filepath}")

    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
