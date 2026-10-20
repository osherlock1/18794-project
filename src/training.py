import torch
from tqdm.auto import tqdm


def evaluate_loss(model, criterion, dataloader, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss


def evaluate_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    return accuracy


def train(
    model,
    optimizer,
    criterion,
    trainloader,
    valloader,
    epochs,
    device,
    on_epoch_end=None,
):
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        model.train()

        for inputs, labels in tqdm(trainloader, desc="Training"):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss}")

        train_accuracy = (running_correct / running_total) * 100
        print(f"Epoch {epoch + 1}/{epochs} - Train Accuracy: {train_accuracy:.2f}%")

        val_loss = evaluate_loss(model, criterion, valloader, device)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss}")

        val_accuracy = evaluate_accuracy(model, valloader, device)
        print(f"Epoch {epoch + 1}/{epochs} - Val Accuracy: {val_accuracy:.2f}%")

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_accuracy,
                    "val_loss": val_loss,
                    "val_acc": val_accuracy,
                }
            )

    return train_losses, val_losses
