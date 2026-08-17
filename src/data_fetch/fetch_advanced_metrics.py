#!/usr/bin/env python3
"""
Fetch advanced Statcast metrics from Baseball Savant.
"""

import requests
import csv
import time
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

def fetch_advanced_metrics():
    """Fetch advanced metrics from Baseball Savant."""
    print("Fetching advanced Statcast metrics from Baseball Savant...")

    # Use Baseball Savant search with specific parameters for advanced metrics
    savant_url = "https://baseballsavant.mlb.com/statcast_search/csv"

    params = {
        'hfPT': '',  # All pitch types
        'hfAB': '',
        'hfBBT': '',
        'hfPR': '',
        'hfZ': '',
        'stadium': '',
        'hfBBL': '',
        'hfNewZones': '',
        'hfGT': 'R|2023-04-01|2023-04-07|',  # Date range
        'hfC': '',
        'hfSea': '2023|',
        'hfSit': '',
        'hfOuts': '',
        'opponent': '',
        'pitcher_throws': '',
        'batter_stands': '',
        'hfSA': '',
        'player_type': 'pitcher',
        'hfInfield': '',
        'team': '',
        'position': '',
        'hfOutfield': '',
        'hfRO': '',
        'home_road': '',
        'hfFlag': '',
        'hfPull': '',
        'metric_1': '',
        'hfInn': '',
        'min_pitches': '10',  # Minimum pitches per pitcher
        'min_results': '0',
        'group_by': 'name',
        'sort_col': 'pitches',
        'player_event_sort': 'api_p_release_speed',
        'sort_order': 'desc',
        'min_abs': '0',
        'type': 'details'
    }

    try:
        print("Making request to Baseball Savant...")
        response = requests.get(savant_url, params=params, timeout=120)
        response.raise_for_status()

        content = response.text.strip()
        lines = content.split('\n')

        if len(lines) < 2:
            print("No data returned from Baseball Savant")
            print(f"Response content: {content[:500]}...")
            return None

        # Parse CSV data
        import io
        csv_reader = csv.DictReader(io.StringIO(content))

        data = []
        for row in csv_reader:
            data.append(row)

        print(f"Retrieved {len(data)} pitcher records from Baseball Savant")
        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    print("Fetching advanced Statcast metrics...")
    data = fetch_advanced_metrics()

    if not data:
        print("Failed to retrieve data")
        return

    # Save the data
    output_file = str(_DEFAULT_DATA_DIR / 'advanced_statcast_metrics_2023.csv')
    with open(output_file, 'w', newline='') as f:
        if data:
            fieldnames = sorted(data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    print(f"Saved {len(data)} records to {output_file}")

    # Show available columns
    if data:
        print("\nAvailable columns:")
        for col in sorted(data[0].keys()):
            print(f"  {col}")

        # Show sample data
        print("\nSample pitcher data:")
        sample = data[0]
        key_cols = ['player_name', 'pitches', 'whiff_percent', 'hard_hit_percent', 'xwoba', 'chase_percent', 'groundballs_percent', 'flyballs_percent']
        for col in key_cols:
            if col in sample:
                print(f"  {col}: {sample[col]}")

if __name__ == "__main__":
    main()