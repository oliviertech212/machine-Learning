import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.model import ImageClassifier
from utils.data_loader import get_dataloaders

# ─────────────────────────────────────────
# SETTINGS - change these as you like
# ─────────────────────────────────────────
DATASET_PATH = "dataset/"
BATCH_SIZE   = 32
EPOCHS       = 20
LEARNING_RATE = 0.001
SAVE_PATH    = "models/image_classifier.pth"

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
train_loader, test_loader, class_names = get_dataloaders(DATASET_PATH, BATCH_SIZE)
print(f"Class names: {class_names}\n")

# Build model
model = ImageClassifier(num_classes=len(class_names)).to(device)

# Loss and optimizer
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
train_losses = []
test_losses  = []
test_accuracies = []

print("Starting training...\n")

for epoch in range(EPOCHS):

    # ── Training phase ──
    model.train()
    total_train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)               # forward pass
        loss = loss_fn(predictions, labels)       # measure loss

        optimizer.zero_grad()                     # reset gradients
        loss.backward()                           # calculate fix
        optimizer.step()                          # apply fix

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── Testing phase ──
    model.eval()
    total_test_loss = 0
    correct = 0
    total   = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)
            total_test_loss += loss.item()

            # Count correct predictions
            predicted_classes = torch.argmax(predictions, dim=1)
            correct += (predicted_classes == labels).sum().item()
            total   += labels.size(0)

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy      = (correct / total) * 100
    test_losses.append(avg_test_loss)
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Test Loss: {avg_test_loss:.4f} | "
          f"Accuracy: {accuracy:.1f}%")

# ─────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────
torch.save(model.state_dict(), SAVE_PATH)
print(f"\nModel saved to {SAVE_PATH}")

# ─────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train Loss", color="red")
ax1.plot(test_losses,  label="Test Loss",  color="green")
ax1.set_title("Loss over epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(test_accuracies, color="blue")
ax2.set_title("Accuracy over epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("Training chart saved to training_results.png")
