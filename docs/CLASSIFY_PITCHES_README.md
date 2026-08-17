# Pitch Classification Script

This script classifies individual pitches from Statcast data into TDA-discovered pitch clusters and exports the results to a CSV file.

## Prerequisites

- TDA model file: `models/tda_mapper_model.pkl` (created by `notebooks/TDA_Pitch_Clustering.ipynb`; found automatically by default)
- Python packages: pandas, numpy, sklearn, pybaseball

## Quick Start

### Default Run (Sept 15, 2025)
```bash
python src/tda/classify_pitches_to_csv.py
```

### Classify Specific Date
```bash
python src/tda/classify_pitches_to_csv.py 2025-09-20 2025-09-20
```

### Classify Date Range
```bash
python src/tda/classify_pitches_to_csv.py 2025-09-10 2025-09-20
```

### Specify Output File
```bash
python src/tda/classify_pitches_to_csv.py 2025-09-15 2025-09-15 -o my_pitches.csv
```

### Use Custom Model Path
```bash
python src/tda/classify_pitches_to_csv.py 2025-09-15 2025-09-15 -m /path/to/tda_mapper_model.pkl
```

## Output CSV Columns

The output CSV contains the following columns for each pitch:

### Game Context
- `game_date` - Date of the game
- `pitcher` - Pitcher ID
- `pitcher_1` - Pitcher name
- `batter` - Batter ID
- `home_team` - Home team
- `away_team` - Away team
- `inning` - Inning number
- `outs_when_up` - Number of outs
- `balls` - Ball count
- `strikes` - Strike count

### Pitch Information
- `pitch_type` - Pitch type code (FF, SL, CH, etc.)
- `p_throws` - Pitcher handedness (L/R)
- `type` - Pitch result type
- `des` - Pitch description

### Pitch Characteristics (Raw Data)
- `release_speed` - Pitch velocity (mph)
- `pfx_x` - Horizontal break (inches)
- `pfx_z` - Induced vertical break (inches)
- `release_spin_rate` - Spin rate (RPM)
- `spin_axis` - Spin axis (degrees)
- `spin_axis_clock` - Spin axis (clock position)
- `release_extension` - Release extension (feet)
- `release_pos_x` - Release position X (feet)
- `release_pos_z` - Release position Z (feet)

### Cluster Classification
- `cluster_id` - Mapper node identifier (e.g., `cube15_cluster1`)
- `cluster_size` - Number of pitcher-pitch types in this cluster
- `distance_to_cluster` - Distance to cluster centroid in scaled feature space
- `dominant_pitch_type_in_cluster` - Most common pitch type in cluster
- `cluster_release_speed` - Average release speed of cluster
- `cluster_HB` - Average horizontal break of cluster
- `cluster_IVB` - Average induced vertical break of cluster

## Example Usage & Output

### Run for single day:
```bash
$ python src/tda/classify_pitches_to_csv.py 2025-09-15 2025-09-15
Loading model from /path/to/models/tda_mapper_model.pkl...
Model loaded successfully

Querying Statcast data from 2025-09-15 to 2025-09-15...
Retrieved 2739 pitch records
Preparing 2739 pitches for classification...
Classifying pitches into clusters...
  Note: Dropped 10 pitches with missing data

Saving 2729 classified pitches to /path/to/data/classified_pitches_2025-09-15_2025-09-15.csv...
Successfully saved to /path/to/data/classified_pitches_2025-09-15_2025-09-15.csv

============================================================
CLASSIFICATION SUMMARY
============================================================
Total pitches classified: 2729

Cluster distribution:
cluster_id
cube59_cluster0    345
cube60_cluster0    411
cube61_cluster0    239
...

Pitch types in data:
pitch_type
FF    1026
SL     479
SI     296
CH     251
CU     219
...

Average distance to assigned cluster: 1.4065
============================================================
```

## Notes

- Pitches with missing data are automatically dropped from the output
- LHP data is mirrored to RHP frame before classification for consistency
- All pitch characteristics are scaled using the training scaler before distance calculation
- The script handles missing Statcast columns gracefully
