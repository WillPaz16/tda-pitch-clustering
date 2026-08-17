# xwOBA baseline training

Run a quick CatBoost baseline to predict `xwOBA` from your pitch dataset.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train:

```bash
python train_xwoba.py --data path/to/your_data.csv --target xwOBA --out models/xwoba_baseline.cbm
```

Notes:
- The script will auto-select common pitch features if present. Edit `train_xwoba.py` to customize features.
- For production, replace the random split with a time-based or grouped split.
