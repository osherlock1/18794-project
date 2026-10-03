import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import get_frames, resize_sequence

BASE_PATH = Path("..")
TRAIN_CSV_PATH = BASE_PATH / "data" / "asl-signs" / "train.csv"
CLEANED_FILE_NAME = "test"
CLEANED_SAVE_PATH = BASE_PATH / "data" / "asl-signs" / "cleaned" / CLEANED_FILE_NAME


def main():
    os.makedirs(CLEANED_SAVE_PATH, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    x = np.lib.format.open_memmap(
        CLEANED_SAVE_PATH, mode="w+", dtype=np.float32, shape=(len(train_df), 24, 1086)
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
