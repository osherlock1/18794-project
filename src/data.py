import numpy as np
import pandas as pd


def get_frames(parquet_frame: pd.DataFrame, exclude_types=("face",)) -> np.ndarray:
    parquet_frame = parquet_frame[~parquet_frame["type"].isin(exclude_types)]
    ordered = parquet_frame.sort_values(["frame", "type", "landmark_index"])
    n_frames = ordered["frame"].nunique()

    return ordered[["x", "y"]].to_numpy(dtype="float32").reshape(n_frames, -1)


# NOTE: In order to have the same amount of frames for ever sequence the current solution
# is to just linearly space out of the frames.  This could have a big impact on perforamnce
# and we should keep this in mind
def resize_sequence(sequence, target_frames=24):
    if len(sequence) == 0:
        raise ValueError("Sequence has no frames")

    indicies = np.rint(np.linspace(0, len(sequence) - 1, target_frames)).astype(int)

    return sequence[indicies]


def pad_sequence(sequence, target_frames=64, pad_value=0.0):
    if len(sequence) == 0:
        raise ValueError("Sequence has no frames")

    sequence = sequence[:target_frames]
    length = len(sequence)

    padded = np.full(
        (target_frames, sequence.shape[1]), pad_value, dtype=sequence.dtype
    )
    padded[:length] = sequence

    return padded, length
