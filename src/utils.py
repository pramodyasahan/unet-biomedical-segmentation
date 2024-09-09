import torch
import matplotlib.pyplot as plt


def loss_curves(epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def accuracy_curves(epochs, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()


def compute_dice(predictions, targets, threshold=0.5):
    predictions = (predictions > threshold).float()
    intersection = (predictions * targets).sum()
    dice = 2 * intersection / (predictions.sum() + targets.sum() + 1e-6)
    return dice

