#!/usr/bin/env python3
"""
Fetch real Statcast data using MLB Stats API directly for advanced metrics.
"""

import requests
import csv
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

def fetch_statcast_data_mlb_api(start_date, end_date):
    """Fetch Statcast data using MLB Stats API."""
    print(f"Fetching Statcast data from {start_date} to {end_date}...")

    try:
        # First, get schedule for the date range
        schedule_url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            'sportId': 1,
            'startDate': start_date,
            'endDate': end_date
        }

        print("Getting game schedule...")
        response = requests.get(schedule_url, params=params)
        response.raise_for_status()
        schedule_data = response.json()

        game_pks = []
        for date in schedule_data.get('dates', []):
            for game in date.get('games', []):
                if game.get('status', {}).get('codedGameState') == 'F':  # Only completed games
                    game_pks.append(game['gamePk'])

        print(f"Found {len(game_pks)} completed games")

        if not game_pks:
            print("No completed games found in date range")
            return None

        # Get Statcast data for first few games
        all_pitches = []
        for game_pk in game_pks[:50]:  # Limit to first 50 games for testing
            print(f"Fetching data for game {game_pk}...")
            game_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
            response = requests.get(game_url)
            response.raise_for_status()
            game_data = response.json()

            # Extract pitch data
            for play in game_data.get('liveData', {}).get('plays', {}).get('allPlays', []):
                for play_event in play.get('playEvents', []):
                    if 'pitchData' in play_event:
                        pitch = play_event['pitchData']
                        pitch.update({
                            'game_pk': game_pk,
                            'batter': play.get('matchup', {}).get('batter', {}).get('id'),
                            'pitcher': play.get('matchup', {}).get('pitcher', {}).get('id'),
                            'description': play_event.get('details', {}).get('description', ''),
                            'type': play_event.get('details', {}).get('type', {}).get('description', ''),
                            'result': play.get('result', {}).get('event', ''),
                        })
                        all_pitches.append(pitch)

        print(f"Retrieved {len(all_pitches)} pitches from MLB Stats API")
        return all_pitches

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    # Fetch a full month of data (April 2023)
    start_date = '2023-04-01'
    end_date = '2023-04-30'

    print("Fetching real Statcast data for advanced metrics (Full Month)...")
    data = fetch_statcast_data_mlb_api(start_date, end_date)

    if not data:
        print("Failed to retrieve data")
        return

    # Save the raw data
    output_file = str(_DEFAULT_DATA_DIR / 'real_statcast_data_april_2023.csv')
    with open(output_file, 'w', newline='') as f:
        if data:
            # Collect all unique fieldnames
            fieldnames = set()
            for pitch in data:
                fieldnames.update(pitch.keys())
            fieldnames = sorted(fieldnames)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        for col in ['startSpeed', 'endSpeed', 'strikeZoneTop', 'strikeZoneBottom', 'description']:
            if col in sample:
                print(f"  {col}: {sample[col]}")

if __name__ == "__main__":
    main()