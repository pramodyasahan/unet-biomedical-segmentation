import torch
import os
from torch import nn, optim
from utils import compute_dice, loss_curves, accuracy_curves
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNetWithSelfAttention
from dataloader import Dataloader


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    running_dice = 0

    for img, mask in tqdm(dataloader, desc="Training"):
        img, mask = img.float().to(device), mask.float().to(device)
        optimizer.zero_grad()

        y_pred = model(img)
        loss = criterion(y_pred, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += compute_dice(y_pred, mask).item()

    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)

    return avg_loss, avg_dice


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    running_dice = 0

    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc="Validation"):
            img, mask = img.float().to(device), mask.float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)

            running_loss += loss.item()
            running_dice += compute_dice(y_pred, mask).item()

    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)

    return avg_loss, avg_dice


if __name__ == '__main__':
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 10
    DATA_DIR = 'data/'

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'trained_models', 'unet2.pth')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_dataset = Dataloader(DATA_DIR, test=False)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetWithSelfAttention(in_channels=3, n_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}\n")

        train_loss, train_dice = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_dice = validate_one_epoch(model, val_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print("-" * 50)
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}  |")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}        |")
        print("-" * 50)

    loss_curves(EPOCHS, train_losses, val_losses)
    accuracy_curves(EPOCHS, train_dices, val_dices)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    if os.path.isfile(MODEL_SAVE_PATH):
        print(f"Model successfully saved to {MODEL_SAVE_PATH}")
    else:
        print(f"Failed to save the model. File does not exist at {MODEL_SAVE_PATH}")

