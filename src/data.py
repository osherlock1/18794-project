import numpy as np
import pandas as pd


def get_frames(parquet_frame: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for _, frame_df in parquet_frame.groupby("frame", sort=True):
        ordered = frame_df.sort_values(["type", "landmark_index"])
        features = ordered[["x", "y"]].to_numpy(dtype="float32").reshape(-1)
        frames.append(features)

    return np.stack(frames)


# NOTE: In order to have the same amount of frames for ever sequence the current solution
# is to just linearly space out of the frames.  This could have a big impact on perforamnce
# and we should keep this in mind
def resize_sequence(sequence, target_frames=24):
    if len(sequence) == 0:
        raise ValueError("Sequence has no frames")

    indicies = np.rint(np.linspace(0, len(sequence) - 1, target_frames)).astype(int)

    return sequence[indicies]
