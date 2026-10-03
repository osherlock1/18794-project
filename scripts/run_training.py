from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import ASLDataset
from src.models import CNNBase
from src.training import train

BATCH_SIZE = 64
LR = 10**-2
EPOCHS = 30

BASE_PATH = Path()
DATASET_PATH = BASE_PATH / "data" / "asl-signs" / "cleaned"


def main():
    dataset = ASLDataset(DATASET_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_size = int(0.9 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = CNNBase()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, val_loss = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        trainloader=train_loader,
        valloader=val_loader,
        epochs=EPOCHS,
        device=device,
    )


if __name__ == "__main__":
    main()
