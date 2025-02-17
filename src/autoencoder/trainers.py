import torch
import matplotlib.pyplot as plt
import os

def train_model(model, optimizer, loss_fn, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        reconstructed, _ = model(data)
        loss = loss_fn(reconstructed, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_model(model, loss_fn, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            reconstructed, _ = model(data)
            loss = loss_fn(reconstructed, data)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def plot_losses(train_loss_history, val_loss_history, plot_filename, clip=True, display=True):
    if clip:
        train_loss_history, val_loss_history = train_loss_history[1:], val_loss_history[1:]

    plt.figure(figsize=(15,8))
    
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 
                 label="Training Loss", color="teal")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 
                 label="Validation Loss", color="salmon")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename)

    if display:
        plt.show()