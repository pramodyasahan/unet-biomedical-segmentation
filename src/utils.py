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


def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 5 * num_samples))

    with torch.no_grad():
        for i, (img, mask) in enumerate(dataloader):
            if i >= num_samples:
                break

            img = img.to(device)
            y_pred = torch.sigmoid(model(img)).cpu().numpy()
            mask = mask.numpy()

            axs[i, 0].imshow(img[0].permute(1, 2, 0))
            axs[i, 0].set_title("Input Image")
            axs[i, 1].imshow(mask[0], cmap='gray')
            axs[i, 1].set_title("Ground Truth Mask")
            axs[i, 2].imshow(y_pred[0, 0], cmap='gray')
            axs[i, 2].set_title("Predicted Mask")

    plt.tight_layout()
    plt.show()


def calculate_accuracy(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    y_pred_binary = (y_pred > 0.5).float()
    correct = (y_pred_binary == y_true).float()

    accuracy = correct.sum() / correct.numel()

    return accuracy
