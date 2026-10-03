import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import get_frames, resize_sequence

BASE_PATH = Path()
DATASET_PATH = BASE_PATH / "data" / "asl-signs"
TRAIN_CSV_PATH = DATASET_PATH / "train.csv"
CLEANED_FILE_NAME = "data.npy"
CLEANED_SAVE_PATH = BASE_PATH / "data" / "asl-signs" / "cleaned"


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
        shape=(len(train_df), 24, 1086),
    )

    print("Data cleaning started...")
    for i, (path, participant_id, sequence_id, sign) in enumerate(
        train_df.itertuples(index=False)
    ):
        file_path = BASE_PATH / "data" / "asl-signs" / path
        p_df = pd.read_parquet(file_path)
        frame_data = get_frames(p_df)
        sequence = resize_sequence(frame_data)

        if sequence.shape != (24, 1086):
            raise ValueError(f"Unexpected sequence shape at row{i}")

        x[i] = sequence

        if i % 1000 == 0:
            print(f"{i}/{len(train_df)}")

    x.flush()
    print("Data cleaning finished!")
    print(x.shape)


if __name__ == "__main__":
    main()
