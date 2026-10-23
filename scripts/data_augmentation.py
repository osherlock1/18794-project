import warnings

import numpy as np

from src.data import resize_sequence

# Landmark layout produced by src.data.get_frames with the face removed: types are
# sorted alphabetically (left_hand, pose, right_hand), then by landmark_index
LEFT_HAND = slice(0, 21)
POSE = slice(21, 54)
RIGHT_HAND = slice(54, 75)
# left shoulder is 11 and right shoulder is 12, shifted by 21 because pose block starts
# at index 21
SHOULDERS = [21 + 11, 21 + 12]

# Checks if hand was detected
def hand_present(clip, hand):
    return ~np.isnan(clip[:, hand, 0]).all(axis=1)

# Returns whichever hand is detected in more frames
def dominant_hand(clip):
    right = hand_present(clip, RIGHT_HAND).sum()
    left = hand_present(clip, LEFT_HAND).sum()
    return RIGHT_HAND if right >= left else LEFT_HAND

# Removes frames where neither hand is visible
def drop_no_hand(clip):
    # remove frames where neither hand was detected
    present = hand_present(clip, LEFT_HAND) | hand_present(clip, RIGHT_HAND)
    if not present.any():
        return clip
    return clip[present]

def normalize(clip):
    # center on the mid-shoulder point and scale to shoulder width
    with warnings.catch_warnings():
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

# Blanks one stretch of frames by setting them to NaN
def random_time_mask(clip, rng, max_frac=0.2):
    T = len(clip)
    max_span = int(T * max_frac)
    # clip too short to mask if under 5 frames
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

    # rotation + scale + shift
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
    # resample long clips down to n_frames and pad short ones at the end
    if len(clip) > n_frames:
        return resize_sequence(clip, n_frames)
    pad = np.full((n_frames - len(clip), *clip.shape[1:]), np.nan, dtype=clip.dtype)
    return np.concatenate([clip, pad])


class ClipTransform:
    """Normalizes a (T, 150) sequence"""

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
