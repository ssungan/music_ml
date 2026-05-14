import pandas as pd
import numpy as np
import os


def load_and_clean(csv_path):
    """
    Shared loading and cleaning pipeline for music_genre.csv.

    Steps:
    1. Remove rows with bad values (NaN, '?', -1, '-1', empty strings) across ALL columns.
    2. Coerce 'tempo' to float, drop remaining NaN rows.
    3. Remove exact full-row duplicates.
    4. Remove conflicting duplicates: same (artist_name, track_name) with different
       categorical labels (e.g., same song assigned two different genres or modes).
    5. Drop metadata columns: artist_name, track_name, instance_id, obtained_date.

    Returns a cleaned DataFrame with 'mode' and 'music_genre' still as strings.
    """
    df = pd.read_csv(csv_path)

    bad_values = ["", " ", "?", -1, "-1"]
    mask = df.isna() | df.isin(bad_values)
    df = df[~mask.any(axis=1)].reset_index(drop=True)
    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df = df.dropna(subset=["tempo"]).reset_index(drop=True)

    df = df.drop_duplicates().reset_index(drop=True)

    key_cols = ["artist_name", "track_name"]
    non_num = df.select_dtypes(exclude="number").columns.difference(key_cols)
    conflict = (
        df.groupby(key_cols)[non_num]
        .transform(lambda c: c.nunique(dropna=False) > 1)
        .any(axis=1)
    )
    df = df[~conflict].reset_index(drop=True)

    # Drop metadata columns if present
    for col in ["artist_name", "track_name", "instance_id", "obtained_date"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df
