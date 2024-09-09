import torch
from torch import nn, optim
from utils import calculate_accuracy
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNetWithSelfAttention
from dataloader import Dataloader

if __name__ == '__main__':
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8
    EPOCHS = 2
    DATA_DIR = 'data/'
    MODEL_SAVE_PATH = '/unet.pth'

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    dice_coefficients = []

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

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_accuracy = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            accuracy = calculate_accuracy(y_pred, mask)
            train_running_accuracy += accuracy.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / len(train_dataloader)
        train_accuracy = train_running_accuracy / len(train_dataloader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_running_loss = 0
        val_running_accuracy = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}")):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()

                accuracy = calculate_accuracy(y_pred, mask)
                val_running_accuracy += accuracy.item()

            val_loss = val_running_loss / len(val_dataloader)
            val_accuracy = val_running_accuracy / len(val_dataloader)

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        print("-" * 30)
        print(
            f"Train Loss Epoch {epoch + 1} : {train_loss:.4f} | Train Accuracy Epoch {epoch + 1} : {train_accuracy:.4f}")
        print(f"Val Loss Epoch {epoch + 1} : {val_loss:.4f} | Val Accuracy Epoch {epoch + 1} : {val_accuracy:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
