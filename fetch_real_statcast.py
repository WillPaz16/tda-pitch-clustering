#!/usr/bin/env python3
"""
Fetch real Statcast data using MLB Stats API directly for advanced metrics.
"""

import requests
import json
import csv
import time
from datetime import datetime, timedelta

def fetch_statcast_data_mlb_api(start_date, end_date):
    """Fetch Statcast data using a different approach."""
    print(f"Fetching Statcast data from {start_date} to {end_date}...")

    # Try using the pybaseball statcast function directly by importing only what we need
def fetch_statcast_data_mlb_api(start_date, end_date):
    """Fetch Statcast data using direct API calls."""
    print(f"Fetching Statcast data from {start_date} to {end_date}...")

    try:
        # Use Baseball Savant CSV export with simpler parameters
        savant_url = "https://baseballsavant.mlb.com/statcast_search/csv"

        params = {
            'all': 'true',  # Get all pitches
            'hfPT': '',  # All pitch types
            'hfAB': '',
            'hfBBT': '',
            'hfPR': '',
            'hfZ': '',
            'stadium': '',
            'hfBBL': '',
            'hfNewZones': '',
            'hfGT': f'R%7C%7C{start_date}%7C{end_date}%7C',
            'hfC': '',
            'hfSea': '2024%7C',
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
            'min_pitches': '0',
            'min_results': '0',
            'group_by': 'name',
            'sort_col': 'pitches',
            'player_event_sort': 'api_p_release_speed',
            'sort_order': 'desc',
            'min_abs': '0',
            'type': 'details'
        }

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

        print(f"Retrieved {len(data)} pitches from Baseball Savant")
        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
def main():
    # Try 2023 season data instead of 2024
    start_date = '2023-04-01'
    end_date = '2023-04-07'

    print("Fetching real Statcast data for advanced metrics...")
    data = fetch_statcast_data_mlb_api(start_date, end_date)

    if not data:
        print("Failed to retrieve data")
        return

    # Save the raw data
    output_file = 'real_statcast_data_2024.csv'
    with open(output_file, 'w', newline='') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    print(f"Saved {len(data)} pitches to {output_file}")

    # Show sample of available columns
    if data:
        print("\nAvailable columns:")
        for col in sorted(data[0].keys())[:20]:  # Show first 20 columns
            print(f"  {col}")

        print(f"\n... and {len(data[0].keys()) - 20} more columns")

        # Show sample data
        print("\nSample pitch data:")
        sample = data[0]
        for col in ['pitch_type', 'launch_speed', 'launch_angle', 'estimated_woba_using_speedangle', 'zone', 'bb_type', 'description']:
            if col in sample:
                print(f"  {col}: {sample[col]}")

if __name__ == "__main__":
    main()