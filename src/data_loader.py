"""
data_loader.py
--------------
Loads all PSV files from a folder and assembles a combined DataFrame.
Each file is one patient; patient_id is derived from the filename.
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_psv_folder(folder_path: str) -> pd.DataFrame:
    """
    Reads every *.psv file in folder_path and returns a single DataFrame.
    Adds a 'patient_id' column derived from the filename.

    Parameters
    ----------
    folder_path : str
        Path to directory containing PSV patient files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all patients, including 'patient_id'.
    """
    psv_files = sorted(glob.glob(os.path.join(folder_path, "*.psv")))
    if not psv_files:
        raise FileNotFoundError(f"No PSV files found in: {folder_path}")

    frames = []
    for fpath in tqdm(psv_files, desc=f"Loading {os.path.basename(folder_path)}"):
        patient_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            df = pd.read_csv(fpath, sep="|", na_values=["NaN", "nan", ""])
            df.insert(0, "patient_id", patient_id)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(psv_files)} patients | Total rows: {len(combined):,}")
    return combined


def load_all_data(data_root: str):
    """
    Convenience wrapper that loads Train_Data and Test_Data sub-folders.

    Parameters
    ----------
    data_root : str
        Root folder containing 'Train_Data' and 'Test_Data' sub-directories.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_path = os.path.join(data_root, "Train_Data")
    test_path  = os.path.join(data_root, "Test_Data")

    print("=== Loading Training Data ===")
    train_df = load_psv_folder(train_path)

    print("\n=== Loading Test Data ===")
    test_df = load_psv_folder(test_path)

    return train_df, test_df
