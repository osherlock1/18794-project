import argparse
import shutil
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "asl-signs"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "data" / COMPETITION
MIN_FREE_GB = 100


def get_api():
    api = KaggleApi()
    api.authenticate()
    return api


def check_disk_space(dest, needed_gb):
    free_gb = shutil.disk_usage(dest).free / 1e9
    if free_gb < needed_gb:
        sys.exit(f"Need {needed_gb} GB free only {free_gb:.1f} GB available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEST, help="where to put the data")

    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    check_disk_space(dest, MIN_FREE_GB)
    api = get_api()
    print(f"Downloading {COMPETITION} to {dest}")
    api.competition_download_files(COMPETITION, path=str(dest), quiet=False)


if __name__ == "__main__":
    main()
