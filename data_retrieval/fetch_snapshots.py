import requests
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class WaybackRateLimiter:
    def __init__(self, requests_per_minute=4):  # Reduced from 15 to be more conservative
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0

    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            sleep_time *= (1 + random.uniform(0, 0.2))  # Only positive jitter
            time.sleep(sleep_time)
        self.last_request_time = time.time()

def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,  # Increased retries
        backoff_factor=2,  # Increased backoff
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

def get_available_snapshots(session: requests.Session,
                          rate_limiter: WaybackRateLimiter,
                          url: str,
                          start_date: datetime,
                          end_date: datetime) -> List[Dict[str, Any]]:
    """
    Get a list of all available snapshots between start_date and end_date.
    Returns list of dictionaries containing timestamp and other metadata.
    """
    cdx_api_url = "http://web.archive.org/cdx/search/cdx"

    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    params = {
        'url': url,
        'matchType': 'exact',
        'from': start_date.strftime("%Y%m%d"),
        'to': end_date.strftime("%Y%m%d"),
        'output': 'json',
        'fl': 'timestamp,statuscode,original',
        'filter': 'statuscode:200',
        'collapse': 'timestamp:8'  # Still get daily snapshots initially
    }

    rate_limiter.wait()
    try:
        response = session.get(cdx_api_url, params=params, timeout=30)
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response content: {response.text[:200]}")
            return []

        if not data or len(data) < 2:
            print(f"No data found in response. Response content: {response.text[:200]}")
            return []

        headers = data[0]
        snapshots = []
        for row in data[1:]:
            snapshot = dict(zip(headers, row))
            snapshots.append(snapshot)

        print(f"Found {len(snapshots)} available snapshots")
        return snapshots

    except requests.exceptions.RequestException as e:
        print(f"Error getting snapshot list: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text[:200]}")
        return []

def filter_weekly_snapshots(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter snapshots to get one per week, preferring snapshots from the middle of each week.
    """
    if not snapshots:
        return []

    weekly_snapshots = {}

    for snapshot in snapshots:
        timestamp = snapshot['timestamp']
        date = datetime.strptime(timestamp[:8], "%Y%m%d")
        # Get the week number and year for this date
        year_week = (date.year, date.isocalendar()[1])

        if year_week not in weekly_snapshots:
            weekly_snapshots[year_week] = snapshot
        else:
            # If we already have a snapshot for this week,
            # prefer the one closer to Wednesday (middle of the week)
            existing_date = datetime.strptime(weekly_snapshots[year_week]['timestamp'][:8], "%Y%m%d")
            existing_weekday = existing_date.weekday()
            current_weekday = date.weekday()

            # Calculate distance from Wednesday (3)
            existing_distance = abs(3 - existing_weekday)
            current_distance = abs(3 - current_weekday)

            if current_distance < existing_distance:
                weekly_snapshots[year_week] = snapshot

    # Convert back to list and sort by timestamp
    result = list(weekly_snapshots.values())
    result.sort(key=lambda x: x['timestamp'])

    print(f"Filtered to {len(result)} weekly snapshots")
    return result

def get_snapshot_content(session: requests.Session,
                        rate_limiter: WaybackRateLimiter,
                        url: str,
                        timestamp: str) -> Optional[str]:
    """
    Get the content of a specific snapshot using its timestamp.
    """
    wayback_url = f"https://web.archive.org/web/{timestamp}id_/{url}"  # Added id_ flag
    print(f"Fetching snapshot: {wayback_url}")

    try:
        rate_limiter.wait()
        response = session.get(wayback_url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching snapshot: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            print("Rate limit hit, waiting 120 seconds...")  # Increased wait time
            time.sleep(120)
        return None

def get_efficient_snapshots(url: str,
                          start_date: datetime,
                          end_date: datetime,
                          output_dir: str,
                          requests_per_minute: int = 4) -> None:
    """
    More efficient version that gets weekly snapshots.
    """
    session = create_session()
    rate_limiter = WaybackRateLimiter(requests_per_minute)

    print(f"Getting list of available snapshots for {url}...")
    all_snapshots = get_available_snapshots(session, rate_limiter, url, start_date, end_date)

    if not all_snapshots:
        print("No snapshots found for the specified date range.")
        return

    # Filter to weekly snapshots
    weekly_snapshots = filter_weekly_snapshots(all_snapshots)
    print(f"Processing {len(weekly_snapshots)} weekly snapshots...")

    for snapshot in weekly_snapshots:
        timestamp = snapshot['timestamp']
        snapshot_date = datetime.strptime(timestamp[:8], "%Y%m%d")
        content = get_snapshot_content(session, rate_limiter, url, timestamp)

        if content:
            save_snapshot(content, url, snapshot_date, output_dir)
            print(f"Successfully saved snapshot from {snapshot_date}")
        else:
            print(f"Failed to retrieve content for snapshot from {snapshot_date}")

def save_snapshot(content: str, url: str, date: datetime, output_dir: str) -> None:
    # Remove protocol and clean URL for directory name
    site_dir = url.replace('http://', '').replace('https://', '').rstrip('/')
    site_dir = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in site_dir)

    full_path = Path(output_dir) / site_dir / str(date.year) / f"{date.month:02d}"
    full_path.mkdir(parents=True, exist_ok=True)

    filename = f"{date.strftime('%Y%m%d')}.html"
    file_path = full_path / filename

    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        print(f"Saved snapshot to: {file_path}")
    except IOError as e:
        print(f"Error saving snapshot: {e}")

def main():
    url = "pornhub.com"
    start_date = datetime(2007, 1, 1)
    end_date = datetime(2024, 1, 1)
    output_dir = "../snapshots"

    print(f"Retrieving snapshots for {url}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Output directory: {output_dir}")

    get_efficient_snapshots(url, start_date, end_date, output_dir)

if __name__ == "__main__":
    main()
