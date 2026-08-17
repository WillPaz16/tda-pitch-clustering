"""
Stuff+ Calculator
Loads trained models and standardization metadata, applies to new statcast data
to calculate pitcher-level Stuff+ metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Default settings
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = str(_PROJECT_ROOT / 'models')
DEFAULT_MAPPINGS_DIR = str(_PROJECT_ROOT / 'data' / 'mappings')
DEFAULT_MIN_PITCHES_PER_TYPE = 30
DEFAULT_MIN_PITCHES_TOTAL = 50


class StuffPlusCalculator:
    """
    Load trained models and calculate Stuff+ for new data.
    
    Usage:
        calc = StuffPlusCalculator(models_dir='models')
        results = calc.calculate(df)
        pitcher_level = results['pitcher_level']
        pitcher_level.to_csv('stuff_plus_results.csv', index=False)
    """
    
    def __init__(self, models_dir=DEFAULT_MODELS_DIR, mappings_dir=DEFAULT_MAPPINGS_DIR):
        """
        Initialize calculator by loading models and metadata.
        
        Args:
            models_dir: Directory containing trained models and metadata
            mappings_dir: Directory containing feature mappings
        """
        self.models_dir = models_dir
        self.mappings_dir = mappings_dir
        self.models = {}
        self.metadata = {}
        self.pitch_type_map = {}
        self.stuff_plus_metadata = {}
        
        self._load_models()
        self._load_metadata()
        self._load_pitch_type_map()
        self._load_stuff_plus_metadata()
    
    def _load_models(self):
        """Load xwOBA, miss, and chase models."""
        for model_name in ['xwoba', 'miss', 'chase']:
            self.models[model_name] = self._load_model(model_name)
            if self.models[model_name] is None:
                print(f"⚠ Warning: {model_name} model not found")
    
    def _load_model(self, name):
        """Load a single model (CatBoost or joblib pickle)."""
        meta_path = os.path.join(self.models_dir, f'{name}_model_metadata.json')
        if not os.path.exists(meta_path):
            return None
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        mtype = meta.get('model_type', '')
        
        # Try CatBoost first
        if 'CatBoost' in mtype:
            try:
                from catboost import CatBoostRegressor, CatBoostClassifier
                cbm_path = os.path.join(self.models_dir, f'{name}_model.cbm')
                if os.path.exists(cbm_path):
                    if 'Regressor' in mtype:
                        m = CatBoostRegressor()
                    else:
                        m = CatBoostClassifier()
                    m.load_model(cbm_path)
                    self.metadata[name] = meta
                    return m
            except Exception as e:
                print(f"  CatBoost load failed for {name}: {e}")
        
        # Fall back to joblib pickle
        pkl_path = os.path.join(self.models_dir, f'{name}_model.pkl')
        if os.path.exists(pkl_path):
            try:
                m = joblib.load(pkl_path)
                self.metadata[name] = meta
                return m
            except Exception as e:
                print(f"  Joblib load failed for {name}: {e}")
        
        return None
    
    def _load_metadata(self):
        """Load model metadata (features, metrics, etc)."""
        for model_name in ['xwoba', 'miss', 'chase']:
            meta_path = os.path.join(self.models_dir, f'{model_name}_model_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.metadata[model_name] = json.load(f)
    
    def _load_pitch_type_map(self):
        """Load pitch_type encoding map."""
        map_path = os.path.join(self.mappings_dir, 'pitch_type_map.json')
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                self.pitch_type_map = json.load(f)
    
    def _load_stuff_plus_metadata(self):
        """Load Stuff+ standardization metadata (league stats, scaling params)."""
        meta_path = os.path.join(self.models_dir, 'stuff_plus_per_pitch_type_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.stuff_plus_metadata = json.load(f)
    
    def _build_features(self, df, stuff_cols=None):
        """
        Engineer features from raw statcast data.
        
        Args:
            df: DataFrame with raw statcast data
            stuff_cols: List of base columns (if None, uses standard stuffCols)
        
        Returns:
            X: Feature matrix (numeric only)
        """
        if stuff_cols is None:
            stuff_cols = [
                'release_speed', 'release_pos_x', 'release_pos_z',
                'pfx_x', 'pfx_z', 'spin_axis',
                'release_spin_rate', 'release_extension'
            ]
        
        # Select available columns
        extra_possible = ['px', 'pz', 'release_speed', 'pfx_x', 'pfx_z',
                         'spin_axis', 'release_pos_x', 'release_pos_z', 'release_extension']
        use_cols = [c for c in stuff_cols + extra_possible if c in df.columns]
        
        work = df[use_cols].copy()
        work = work.loc[:, ~work.columns.duplicated()]
        
        # Coerce to numeric
        for c in work.columns:
            col = work[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
                work[c] = col
            if col.dtype != 'object':
                work[c] = pd.to_numeric(col, errors='coerce')
        
        # Speed transforms
        if 'release_speed' in work.columns:
            work['release_speed_sq'] = work['release_speed'] ** 2
            work['log_release_speed'] = np.log1p(work['release_speed'])
        
        # Movement magnitude
        if 'pfx_x' in work.columns or 'pfx_z' in work.columns:
            px_val = work.get('pfx_x', pd.Series(0, index=work.index)).fillna(0)
            pz_val = work.get('pfx_z', pd.Series(0, index=work.index)).fillna(0)
            work['pfx_magnitude'] = np.sqrt(px_val**2 + pz_val**2)
        
        # Spin components
        if 'spin_axis' in work.columns:
            work['spin_axis_rad'] = np.deg2rad(work['spin_axis'].astype(float))
            work['spin_cos'] = np.cos(work['spin_axis_rad'])
            work['spin_sin'] = np.sin(work['spin_axis_rad'])
        
        # Release interactions
        if 'release_pos_x' in work.columns and 'release_pos_z' in work.columns:
            work['release_side'] = work['release_pos_x']
            work['release_height'] = work['release_pos_z']
            work['release_side_times_height'] = work['release_pos_x'] * work['release_pos_z']
        
        # Location proxy
        pz_center = work['pz'].median() if 'pz' in work.columns and work['pz'].notna().any() else 2.5
        if 'pz' in work.columns:
            work['dist_zone_center'] = np.sqrt(work.get('px', 0)**2 + (work['pz'] - pz_center)**2)
        
        # Pitch type encoding
        if 'pitch_type' in work.columns:
            codes, uniques = pd.factorize(work['pitch_type'].astype(str))
            work['pitch_type_code'] = codes
        
        # Drop IDs
        for col in ['pitcher_id', 'batter_id', 'umpire_id', 'catcher_id']:
            if col in work.columns:
                work.drop(columns=[col], inplace=True, errors='ignore')
        
        # Get numeric features
        feat_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
        
        # Impute and add missing indicators
        for c in feat_cols:
            if work[c].isna().any():
                med = work[c].median()
                work[c + '_missing'] = work[c].isna().astype(int)
                work[c] = work[c].fillna(med)
        
        feat_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
        return work[feat_cols]
    
    def _predict_safe(self, model, X):
        """Safely generate predictions (handles both regressors and classifiers)."""
        if model is None:
            return np.zeros(len(X))
        try:
            if hasattr(model, 'predict_proba'):
                return np.array(model.predict_proba(X)[:, 1], dtype=float)
            else:
                return np.array(model.predict(X), dtype=float)
        except Exception as e:
            print(f"  Prediction failed: {e}")
            return np.zeros(len(X))
    
    def calculate(self, df, min_pitches_per_type=DEFAULT_MIN_PITCHES_PER_TYPE,
                  min_pitches_total=DEFAULT_MIN_PITCHES_TOTAL):
        """
        Calculate Stuff+ for a dataframe.
        
        Args:
            df: Statcast DataFrame with columns: pitcher_id, pitch_type, and pitch features
            min_pitches_per_type: Minimum pitches per pitch-type for stability (default: 30)
            min_pitches_total: Minimum total pitches for pitcher stability (default: 50)
        
        Returns:
            Dictionary with keys:
                - 'pitch_level': Full df with all predictions and Stuff+ metrics
                - 'per_pitch_type': Aggregated to pitcher × pitch_type
                - 'pitcher_level': Pitcher-level Stuff+ rankings with names
        """
        
        # Validate required columns
        if 'pitcher_id' not in df.columns or 'pitch_type' not in df.columns:
            raise ValueError("df must contain 'pitcher_id' and 'pitch_type' columns")
        
        print("Building features...")
        X = self._build_features(df)
        print(f"  Built {X.shape[1]} features from {len(X)} pitches")
        
        # Generate predictions
        print("Generating predictions...")
        pred_xw = self._predict_safe(self.models.get('xwoba'), X)
        pred_miss = self._predict_safe(self.models.get('miss'), X)
        pred_chase = self._predict_safe(self.models.get('chase'), X)
        print(f"  xwOBA: {pred_xw.sum():.1f} total, miss: {pred_miss.sum():.1f}, chase: {pred_chase.sum():.1f}")
        
        # Align predictions back to df
        df_result = df.iloc[:len(X)].copy()
        df_result['pred_xw'] = pred_xw
        df_result['pred_miss'] = pred_miss
        df_result['pred_chase'] = pred_chase
        
        # Aggregate to pitcher-pitch_type
        print("Aggregating to pitcher-pitch_type...")
        agg_pt = df_result.groupby(['pitcher_id', 'pitch_type']).agg(
            pitch_count_pt=('pred_xw', 'count'),
            mean_xw_pt=('pred_xw', 'mean'),
            mean_miss_pt=('pred_miss', 'mean'),
            mean_chase_pt=('pred_chase', 'mean')
        ).reset_index()
        
        # Load or compute league stats
        pt_stats = self.stuff_plus_metadata.get('pitch_type_stats', {})
        
        if not pt_stats:
            print("  Computing league stats from data (no cached stats found)...")
            for pt, grp in agg_pt.groupby('pitch_type'):
                stable_grp = grp[grp['pitch_count_pt'] >= min_pitches_per_type]
                if len(stable_grp) < 10:
                    stable_grp = grp
                
                mu_xw = stable_grp['mean_xw_pt'].mean()
                sd_xw = stable_grp['mean_xw_pt'].std(ddof=0)
                mu_miss = stable_grp['mean_miss_pt'].mean()
                sd_miss = stable_grp['mean_miss_pt'].std(ddof=0)
                mu_chase = stable_grp['mean_chase_pt'].mean()
                sd_chase = stable_grp['mean_chase_pt'].std(ddof=0)
                
                pt_stats[pt] = {
                    'mu_xw': float(mu_xw), 'sd_xw': float(max(sd_xw, 0.01)),
                    'mu_miss': float(mu_miss), 'sd_miss': float(max(sd_miss, 0.01)),
                    'mu_chase': float(mu_chase), 'sd_chase': float(max(sd_chase, 0.01)),
                    'n_stable': int(len(stable_grp))
                }
        
        # Compute z-scores per pitcher-pitch_type
        print("Computing z-scores and Stuff+ index...")
        def compute_z_row(row):
            pt = row['pitch_type']
            stats = pt_stats.get(pt)
            if stats is None:
                zx = -(row['mean_xw_pt'] - 0.0) / 0.01
                zm = (row['mean_miss_pt'] - 0.0) / 0.01
                zc = (row['mean_chase_pt'] - 0.0) / 0.01
            else:
                zx = -(row['mean_xw_pt'] - stats['mu_xw']) / stats['sd_xw']
                zm = (row['mean_miss_pt'] - stats['mu_miss']) / stats['sd_miss']
                zc = (row['mean_chase_pt'] - stats['mu_chase']) / stats['sd_chase']
            return pd.Series({'z_xw_pt': zx, 'z_miss_pt': zm, 'z_chase_pt': zc})
        
        z_df = agg_pt.apply(compute_z_row, axis=1)
        agg_pt = pd.concat([agg_pt, z_df], axis=1)
        
        # Get weights (default to equal if not cached)
        weights = np.array(self.stuff_plus_metadata.get('weights_used', [1/3, 1/3, 1/3]), dtype=float)
        
        # Combine into stuff_z
        agg_pt['stuff_z_pt'] = weights[0]*agg_pt['z_xw_pt'] + weights[1]*agg_pt['z_miss_pt'] + weights[2]*agg_pt['z_chase_pt']
        
        # Scale per pitch_type
        agg_pt['stuff_plus_pt'] = np.nan
        target_mean, target_sd = 100.0, 15.0
        for pt, grp in agg_pt.groupby('pitch_type'):
            stable_grp = grp[grp['pitch_count_pt'] >= min_pitches_per_type]
            if len(stable_grp) < 10:
                stable_grp = grp
            mean_z = stable_grp['stuff_z_pt'].mean()
            sd_z = stable_grp['stuff_z_pt'].std(ddof=0)
            sd_z = max(sd_z, 0.01)
            mask = agg_pt['pitch_type'] == pt
            agg_pt.loc[mask, 'stuff_plus_pt'] = target_mean + target_sd * ((agg_pt.loc[mask, 'stuff_z_pt'] - mean_z) / sd_z)
        
        # Aggregate to pitcher-level
        pitch_total = agg_pt.groupby('pitcher_id')['pitch_count_pt'].sum().rename('total_pitches').reset_index()
        agg_pt = agg_pt.merge(pitch_total, on='pitcher_id', how='left')
        
        agg_pitch = agg_pt.groupby('pitcher_id').apply(
            lambda g: pd.Series({
                'total_pitches': int(g['total_pitches'].iloc[0]),
                'weighted_stuff_z': np.average(g['stuff_z_pt'], weights=g['pitch_count_pt']) if g['pitch_count_pt'].sum() > 0 else np.nan,
                'weighted_stuff_plus_by_pt': np.average(g['stuff_plus_pt'], weights=g['pitch_count_pt']) if g['pitch_count_pt'].sum() > 0 else np.nan
            })
        ).reset_index()
        
        # Final scaling
        stable_pitchers = agg_pitch[agg_pitch['total_pitches'] >= min_pitches_total]
        if len(stable_pitchers) < 10:
            stable_pitchers = agg_pitch.copy()
        zs_mean = stable_pitchers['weighted_stuff_z'].mean()
        zs_sd = stable_pitchers['weighted_stuff_z'].std(ddof=0)
        zs_sd = max(zs_sd, 0.01)
        agg_pitch['stuff_plus'] = target_mean + target_sd * ((agg_pitch['weighted_stuff_z'] - zs_mean) / zs_sd)
        agg_pitch['stable'] = agg_pitch['total_pitches'] >= min_pitches_total
        
        # Add pitcher names if available
        if 'pitcher' in df.columns:
            pitcher_names = df.drop_duplicates('pitcher_id')[['pitcher_id', 'pitcher']]
            agg_pitch = agg_pitch.merge(pitcher_names, on='pitcher_id', how='left')
        
        # Merge metrics back to df_result
        df_result = df_result.merge(
            agg_pt[['pitcher_id', 'pitch_type', 'pitch_count_pt', 'mean_xw_pt', 'mean_miss_pt', 'mean_chase_pt', 'stuff_plus_pt']],
            on=['pitcher_id', 'pitch_type'], how='left'
        )
        df_result = df_result.merge(
            agg_pitch[['pitcher_id', 'total_pitches', 'stuff_plus', 'stable']],
            on='pitcher_id', how='left'
        )
        
        # Sort pitcher results by Stuff+
        agg_pitch_sorted = agg_pitch.sort_values('stuff_plus', ascending=False)
        
        print(f"\n✓ Stuff+ calculated for {len(agg_pitch)} pitchers ({stable_pitchers.shape[0]} stable)")
        
        return {
            'pitch_level': df_result,
            'per_pitch_type': agg_pt,
            'pitcher_level': agg_pitch_sorted
        }