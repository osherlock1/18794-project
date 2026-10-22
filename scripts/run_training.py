import argparse
from pathlib import Path

import torch
import wandb
import yaml
from torch.utils.data import DataLoader, random_split

from src.datasets import ASLDataset
from src.models import CNNBase, CNNTransformer
from src.training import evaluate_accuracy, train

BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "data" / "asl-signs" / "cleaned"
CONFIG_PATH = BASE_PATH / "configs"
WEIGHTS_PATH = BASE_PATH / "weights"

MODELS = {"CNNTransformer": CNNTransformer, "CNNBase": CNNBase}

OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

CRITERIONS = {"cross_entropy": torch.nn.CrossEntropyLoss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Name of config file for the training.  Should be a yaml type saved in the configs dir",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH / args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb.init(project="asl-signs", name=args.config.stem, config=config)

    dataset = ASLDataset(DATASET_PATH, sequence_file_name=config["data_file_name"])

    data_frames = dataset.sequences.shape[1]
    if data_frames != config["n_frames"]:
        raise ValueError(
            f"Config n_frames is {config['n_frames']} but {config['data_file_name']} has {data_frames} frames"
        )

    if config["device"] == "mps":
        device = "mps" if torch.mps.is_available() else "cpu"

    elif config["device"] == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif config["device"] == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"

    print(f"Device being used: {device}")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    print(f"Train Size: {train_size}, Val Size: {val_size}, Test Size: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = MODELS[config["model"]](n_frames=config["n_frames"])

    optimizer_config = dict(config["optimizer"])
    optimizer_class = OPTIMIZERS[optimizer_config.pop("name")]
    optimizer = optimizer_class(params=model.parameters(), **optimizer_config)

    criterion_config = dict(config["criterion"])
    criterion_class = CRITERIONS[criterion_config.pop("name")]
    criterion = criterion_class(**criterion_config)

    WEIGHTS_PATH.mkdir(exist_ok=True)
    checkpoint_path = WEIGHTS_PATH / f"{args.config.stem}_{wandb.run.id}_best.pt"
    best_val_acc = 0.0

    def on_epoch_end(metrics):
        nonlocal best_val_acc
        wandb.log(metrics)

        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            torch.save(
                {
                    "epoch": metrics["epoch"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint (val acc {best_val_acc:.2f}%)")

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        trainloader=train_loader,
        valloader=val_loader,
        epochs=config["epochs"],
        device=device,
        on_epoch_end=on_epoch_end,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Test Accuracy (best epoch {checkpoint['epoch']}): {test_accuracy:.2f}%")
    wandb.summary["test_acc"] = test_accuracy
    wandb.summary["best_val_acc"] = checkpoint["val_acc"]
    wandb.summary["best_epoch"] = checkpoint["epoch"]
    wandb.finish()


if __name__ == "__main__":
    main()
