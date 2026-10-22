import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import get_frames, resize_sequence

BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "data" / "asl-signs"
TRAIN_CSV_PATH = DATASET_PATH / "train.csv"
CLEANED_SAVE_PATH = DATASET_PATH / "cleaned"
N_FEATURES = 150


def process_file(path, n_frames):
    p_df = pd.read_parquet(
        DATASET_PATH / path, columns=["frame", "type", "landmark_index", "x", "y"]
    )
    return resize_sequence(get_frames(p_df), n_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-frames",
        "-n",
        type=int,
        required=True,
        help="Number of frames each sequence is resized to",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="File name of the .npy to save the sequences to. Saved in the cleaned dir along with labels.npy",
    )
    args = parser.parse_args()

    n_frames = args.n_frames
    save_dir = CLEANED_SAVE_PATH
    output_path = save_dir / args.output

    os.makedirs(save_dir, exist_ok=True)
    with open(DATASET_PATH / "sign_to_prediction_index_map.json", "r") as f:
        sign_to_index = json.load(f)

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    label_ids = train_df["sign"].map(sign_to_index)
    if label_ids.isna().any():
        raise ValueError("Some signs are missing from the labels map")

    np.save(
        save_dir / "labels.npy",
        label_ids.to_numpy(dtype=np.int64),
    )

    x = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(train_df), n_frames, N_FEATURES),
    )

    print("Data cleaning started...")
    with Pool() as pool:
        results = pool.imap(
            partial(process_file, n_frames=n_frames), train_df["path"], chunksize=64
        )
        for i, sequence in enumerate(results):
            if sequence.shape != (n_frames, N_FEATURES):
                raise ValueError(f"Unexpected sequence shape at row{i}")

            x[i] = sequence

            if i % 1000 == 0:
                print(f"{i}/{len(train_df)}")

    x.flush()
    print("Data cleaning finished!")
    print(x.shape)


if __name__ == "__main__":
    main()
