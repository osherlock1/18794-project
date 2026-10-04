import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/hands_connections.py

# TODO: We need to expand this plot things other than just the hands.
# Face is more difficult because its spread out across face features like =
# Eyes mouth nose etc..

HAND_CONNECTIONS = [
    (0, 1),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
]


# TODO: Hardcoded defaults for now we can
# look to do somekind of centering if we want
X_LIMITS = (0, 1)
Y_LIMITS = (0, 1)


def hand_xy(df: pd.DataFrame, frame_id: int, hand_type: str) -> np.ndarray:
    """
    Helper to get hand x
    """
    hand = (
        df.loc[(df["frame"] == frame_id) & (df["type"] == hand_type)]
        .set_index("landmark_index")
        .reindex(
            range(21)
        )  # TODO: IMPORTANT THIS IS TEMP HARDCODED FOR THE HAND PLACEMENT NEEDS TO GENERALIZED
    )
    return hand[["x", "y"]].to_numpy(dtype=float)


def plot_hand(
    ax: plt.Axes, df: pd.DataFrame, frame_id: int, hand_type: str, label_points=False
):
    """
    Plots a hand frame data from a selected dataframe

    Args:
        ax: Matplotlib Axes object
        df: Dataframe containing hand frame data
        frame_id: Frame id in dataframe
        hand_type: "left_hand" or "right_hand"
        label_points: Option to display the label number on plot
    """
    xy = hand_xy(df=df, frame_id=frame_id, hand_type=hand_type)
    valid = np.isfinite(xy).all(axis=1)

    for start, end in HAND_CONNECTIONS:
        if valid[start] and valid[end]:
            ax.plot(
                xy[[start, end], 0],
                xy[[start, end], 1],
                color="coral",
                linewidth=2,
                zorder=1,
            )

    ax.scatter(xy[valid, 0], xy[valid, 1], s=28, color="teal", zorder=2)

    if label_points:
        for index in np.flatnonzero(valid):
            ax.annotate(
                str(index),
                xy[index],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    if not valid.any():
        ax.text(
            0.5,
            0.5,
            "Hand not detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{hand_type.replace('_', ' ').title()} · frame {frame_id}")
    ax.set_xlabel("Normalized x")
    ax.set_ylabel("Normalized y")
    ax.grid(alpha=0.15)


def visualize_loss(train_losses, test_losses):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses\n")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
