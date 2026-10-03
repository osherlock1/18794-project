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


def train(model, optimizer, criterion, trainloader, valloader, epochs, device):
    train_losses = []
    test_losses = []

    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
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

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss}")

        test_loss = evaluate_loss(model, criterion, valloader, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {test_loss}")

        test_accuracy = evaluate_accuracy(model, valloader, device)
        print(f"Epoch {epoch + 1}/{epochs} - Val Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_losses
