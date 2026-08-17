#!/usr/bin/env python3
"""
Fetch advanced Statcast metrics from Baseball Savant.
"""

import requests
import csv
import time

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
    output_file = 'advanced_statcast_metrics_2023.csv'
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
        'min_abs': '0',
        'type': 'details'
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            print("No data returned")
            return []

        header = lines[0].split(',')
        data = []

        for line in lines[1:]:
            if line.strip():
                values = line.split(',')
                if len(values) >= len(header):
                    row = dict(zip(header, values))
                    data.append(row)

        print(f"Retrieved {len(data)} pitches")
        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def calculate_additional_metrics(pitch_data):
    """Calculate additional metrics."""
    for pitch in pitch_data:
        # Hard hit (launch speed >= 95 mph)
        try:
            launch_speed = float(pitch.get('launch_speed', 0) or 0)
            pitch['hard_hit'] = 1 if launch_speed >= 95 else 0
        except (ValueError, TypeError):
            pitch['hard_hit'] = 0

        # Chase (swung at pitch outside zone)
        try:
            zone = int(pitch.get('zone', 0) or 0)
            swung = 1 if pitch.get('description') in ['swinging_strike', 'foul', 'hit_into_play'] else 0
            outside_zone = 1 if zone not in [1,2,3,4,5,6,7,8,9] else 0
            pitch['chase'] = 1 if (outside_zone and swung) else 0
        except (ValueError, TypeError):
            pitch['chase'] = 0

        # BABIP components
        in_play = 1 if pitch.get('type') == 'X' else 0
        hit = 1 if any(word in pitch.get('des', '').lower() for word in ['single', 'double', 'triple', 'home run']) else 0
        pitch['babip_eligible'] = in_play
        pitch['babip_hit'] = 1 if (in_play and hit) else 0

        # Ground ball / Fly ball
        bb_type = pitch.get('bb_type', '')
        pitch['ground_ball'] = 1 if bb_type == 'ground_ball' else 0
        pitch['fly_ball'] = 1 if bb_type == 'fly_ball' else 0

    return pitch_data

def merge_with_existing_data(statcast_data, existing_csv_path):
    """Merge Statcast metrics with existing data."""
    print(f"Merging with {existing_csv_path}...")

    existing_data = []
    with open(existing_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_data.append(row)

    print(f"Loaded {len(existing_data)} existing pitches")

    # Create lookup by game context
    statcast_lookup = {}
    for pitch in statcast_data:
        game_date = pitch.get('game_date', '')
        pitcher = pitch.get('pitcher', '')
        inning = pitch.get('inning', '')
        balls = pitch.get('balls', '')
        strikes = pitch.get('strikes', '')

        key = f"{game_date}_{pitcher}_{inning}_{balls}_{strikes}"
        if key not in statcast_lookup:
            statcast_lookup[key] = []
        statcast_lookup[key].append(pitch)

    merged_count = 0
    for pitch in existing_data:
        game_date = pitch.get('game_date', '')
        pitcher = pitch.get('pitcher', '')
        inning = pitch.get('inning', '')
        balls = pitch.get('balls', '')
        strikes = pitch.get('strikes', '')

        key = f"{game_date}_{pitcher}_{inning}_{balls}_{strikes}"

        if key in statcast_lookup and statcast_lookup[key]:
            statcast_pitch = statcast_lookup[key].pop(0)

            pitch.update({
                'launch_speed': statcast_pitch.get('launch_speed'),
                'launch_angle': statcast_pitch.get('launch_angle'),
                'xwoba': statcast_pitch.get('estimated_woba_using_speedangle'),
                'zone': statcast_pitch.get('zone'),
                'bb_type': statcast_pitch.get('bb_type'),
                'hard_hit': statcast_pitch.get('hard_hit'),
                'chase': statcast_pitch.get('chase'),
                'babip_eligible': statcast_pitch.get('babip_eligible'),
                'babip_hit': statcast_pitch.get('babip_hit'),
                'ground_ball': statcast_pitch.get('ground_ball'),
                'fly_ball': statcast_pitch.get('fly_ball')
            })
            merged_count += 1

    print(f"Merged {merged_count} pitches")
    return existing_data

def main():
    date_ranges = [
        ('2025-09-15', '2025-09-15'),
        ('2026-04-01', '2026-04-07')
    ]

    all_statcast_data = []
    for start_date, end_date in date_ranges:
        data = fetch_statcast_data_directly(start_date, end_date)
        if data:
            all_statcast_data.extend(data)
        time.sleep(2)

    if not all_statcast_data:
        print("No data retrieved")
        return

    print(f"Total Statcast data: {len(all_statcast_data)} pitches")

    all_statcast_data = calculate_additional_metrics(all_statcast_data)

    csv_files = [
        'classified_pitches_2025-09-15_2025-09-15.csv',
        'classified_pitches_2026-04-01_2026-04-07.csv'
    ]

    for csv_file in csv_files:
        try:
            merged_data = merge_with_existing_data(all_statcast_data.copy(), csv_file)

            output_file = csv_file.replace('.csv', '_enhanced.csv')
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=merged_data[0].keys())
                writer.writeheader()
                writer.writerows(merged_data)
            print(f"Saved to {output_file}")

        except FileNotFoundError:
            print(f"File {csv_file} not found")

if __name__ == "__main__":
    main()