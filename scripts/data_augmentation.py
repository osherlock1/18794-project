import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import resize_sequence

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "asl-signs" / "cleaned"

# Landmark layout produced by src.data.get_frames with the face removed: types are
# sorted alphabetically (left_hand, pose, right_hand), then by landmark_index
LEFT_HAND = slice(0, 21)
POSE = slice(21, 54)
RIGHT_HAND = slice(54, 75)
SHOULDERS = [21 + 11, 21 + 12]  # pose landmarks 11/12 = left/right shoulder


def hand_present(clip, hand):
    return ~np.isnan(clip[:, hand, 0]).all(axis=1)


def dominant_hand(clip):
    right = hand_present(clip, RIGHT_HAND).sum()
    left = hand_present(clip, LEFT_HAND).sum()
    return RIGHT_HAND if right >= left else LEFT_HAND


def drop_no_hand(clip):
    # remove frames where neither hand was detected
    present = hand_present(clip, LEFT_HAND) | hand_present(clip, RIGHT_HAND)
    if not present.any():
        return clip
    return clip[present]


def normalize(clip):
    # center on the mid-shoulder point and scale to shoulder width
    with warnings.catch_warnings():
        # nanmean warns on landmarks that are missing in every frame
        warnings.simplefilter("ignore", RuntimeWarning)
        shoulders = clip[:, SHOULDERS]
        center = np.nanmean(shoulders, axis=(0, 1))
        scale = np.nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1))
        if not np.isfinite(scale) or scale < 1e-3:
            center = np.nanmean(clip, axis=(0, 1))
            scale = np.nanstd(clip)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (clip - center) / scale


def random_time_mask(clip, rng, max_frac=0.2):
    T = len(clip)
    max_span = int(T * max_frac)
    # clip too short to mask (under 5 frames)
    if max_span < 1:
        return clip
    # mask 1: 20% of the clip
    span = rng.integers(1, max_span + 1)
    start = rng.integers(0, T - span + 1)
    clip = clip.copy()
    clip[start : start + span] = np.nan
    return clip


def augment(clip, rng):
    # change speed of clip
    new_T = max(2, int(round(len(clip) * rng.uniform(0.5, 1.5))))
    clip = resize_sequence(clip, new_T)

    if rng.random() < 0.5:
        clip = random_time_mask(clip, rng)

    # rotation + scale + shift (coords are centered, units = shoulder width)
    a, s = np.deg2rad(rng.uniform(-15, 15)), rng.uniform(0.9, 1.1)
    R = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    clip = clip @ R.T + rng.uniform(-0.1, 0.1, size=2).astype(np.float32)

    # occasionally blank the non-dominant hand
    if rng.random() < 0.1:
        other = LEFT_HAND if dominant_hand(clip) == RIGHT_HAND else RIGHT_HAND
        clip = clip.copy()
        clip[:, other] = np.nan
    return clip


def fit_length(clip, n_frames):
    # resample long clips down to n_frames and NaN-pad short ones at the end
    if len(clip) > n_frames:
        return resize_sequence(clip, n_frames)
    pad = np.full((n_frames - len(clip), *clip.shape[1:]), np.nan, dtype=clip.dtype)
    return np.concatenate([clip, pad])


class ClipTransform:
    """Normalizes (and optionally augments) a (T, 150) sequence, keeping its shape."""

    def __init__(self, augment=False, seed=None):
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __call__(self, sequence):
        n_frames = len(sequence)
        clip = sequence.reshape(n_frames, -1, 2)
        clip = drop_no_hand(clip)
        clip = normalize(clip)
        if self.augment:
            clip = augment(clip, self.rng)
        clip = fit_length(clip, n_frames)
        return clip.reshape(n_frames, -1).astype(np.float32)


def plot_frame(ax, sequence, title):
    points = sequence.reshape(-1, 2)
    points = points[~np.isnan(points).any(axis=1)]
    ax.scatter(points[:, 0], points[:, 1], s=5)
    ax.invert_yaxis()  # image coords: y points down
    ax.set_aspect("equal")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(
        description="Preview the training augmentation on one clip"
    )
    parser.add_argument("--data-file", default="sequences_noface.npy")
    parser.add_argument("--index", type=int, default=0, help="clip to preview")
    parser.add_argument("--frame", type=int, default=0, help="frame to plot")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sequences = np.load(DATA_PATH / args.data_file, mmap_mode="r")
    clip = sequences[args.index].astype(np.float32)

    normalized = ClipTransform(augment=False)(clip)
    augmented = ClipTransform(augment=True, seed=args.seed)(clip)

    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_frame(axes[0], clip[args.frame], "raw")
    plot_frame(axes[1], normalized[args.frame], "normalized")
    plot_frame(axes[2], augmented[args.frame], "augmented")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
