import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import get_frames, resize_sequence

BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "data" / "asl-signs"
TRAIN_CSV_PATH = DATASET_PATH / "train.csv"
CLEANED_FILE_NAME = "sequences_noface_128.npy"
CLEANED_SAVE_PATH = BASE_PATH / "data" / "asl-signs" / "cleaned"
N_FEATURES = 150

N_FRAME = 128


def process_file(path):
    p_df = pd.read_parquet(
        DATASET_PATH / path, columns=["frame", "type", "landmark_index", "x", "y"]
    )
    return resize_sequence(get_frames(p_df), N_FRAME)


def main():
    os.makedirs(CLEANED_SAVE_PATH, exist_ok=True)
    with open(DATASET_PATH / "sign_to_prediction_index_map.json", "r") as f:
        sign_to_index = json.load(f)

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    label_ids = train_df["sign"].map(sign_to_index)
    if label_ids.isna().any():
        raise ValueError("Some signs are missing from the labels map")

    np.save(
        CLEANED_SAVE_PATH / "labels.npy",
        label_ids.to_numpy(dtype=np.int64),
    )

    x = np.lib.format.open_memmap(
        CLEANED_SAVE_PATH / CLEANED_FILE_NAME,
        mode="w+",
        dtype=np.float32,
        shape=(len(train_df), N_FRAME, N_FEATURES),
    )

    print("Data cleaning started...")
    with Pool() as pool:
        results = pool.imap(process_file, train_df["path"], chunksize=64)
        for i, sequence in enumerate(results):
            if sequence.shape != (N_FRAME, N_FEATURES):
                raise ValueError(f"Unexpected sequence shape at row{i}")

            x[i] = sequence

            if i % 1000 == 0:
                print(f"{i}/{len(train_df)}")

    x.flush()
    print("Data cleaning finished!")
    print(x.shape)


if __name__ == "__main__":
    main()
