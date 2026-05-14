import pandas as pd
import numpy as np


def load_and_clean(csv_path):
    """
    Shared loading and cleaning pipeline for music_genre.csv.
    Used by both music_modes_prediction and music_classifying_genres notebooks.

    Cleaning steps:
    1. Remove rows with bad values (NaN, '?', -1, '-1', empty strings) across ALL columns.
    2. Coerce 'tempo' to float, drop any remaining NaN tempo rows.
    3. Remove fully identical duplicate rows.
    4. Remove conflicting duplicates: same (artist_name, track_name) pair with
       different categorical labels (e.g., same song assigned two different genres or modes).
    5. Drop metadata columns: artist_name, track_name, instance_id, obtained_date.

    Returns a cleaned DataFrame with 'mode' and 'music_genre' still as strings.
    Feature engineering, encoding, and scaling are notebook-specific and not included here.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded:                      {len(df):>6,} rows, {df.shape[1]} columns")

    # Broad sweep — catches NaN, '?', -1, '-1', empty strings in one pass
    bad_values = ['', ' ', '?', -1, '-1']
    mask = df.isna() | df.isin(bad_values)
    df = df[~mask.any(axis=1)].reset_index(drop=True)
    print(f"After bad value removal:     {len(df):>6,} rows")

    # Coerce tempo to float (handles any residual non-numeric values)
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
    df = df.dropna(subset=['tempo']).reset_index(drop=True)

    # Remove fully identical rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Exact duplicates removed:    {n_before - len(df):>6,}")

    # Remove rows where the same (artist_name, track_name) has conflicting categorical
    # labels — e.g., same song assigned two different genres or two different modes
    key_cols = ['artist_name', 'track_name']
    non_numeric_cols = df.select_dtypes(exclude='number').columns.difference(key_cols)
    conflicting_mask = (
        df.groupby(key_cols)[non_numeric_cols]
          .transform(lambda col: col.nunique(dropna=False) > 1)
          .any(axis=1)
    )
    n_before = len(df)
    df = df[~conflicting_mask].reset_index(drop=True)
    print(f"Conflicting duplicates removed: {n_before - len(df):>4,}")

    # Drop metadata columns not used for modelling
    df = df.drop(columns=['artist_name', 'track_name', 'instance_id', 'obtained_date'])
    print(f"Final shape:                 {df.shape}")
    return df
