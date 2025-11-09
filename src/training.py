import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ CHANGE THESE PATHS
# -------------------------------
train_dir = r"C:\Users\mrans\Desktop\PlantDiseaseAI\datasets\train"  # <-- Change to your train folder
test_dir = r"C:\Users\mrans\Desktop\PlantDiseaseAI\datasets\valid"  # <-- Change to your test folder
save_model_path = r"C:\Users\mrans\Desktop\PlantDiseaseAI\models\mobilenetv2_model.pth"  # <-- Where model will be saved

# -------------------------------
# 3️⃣ Hyperparameters
# -------------------------------
num_epochs = 15
batch_size = 16
learning_rate = 1e-4
input_size = 224
unfreeze_epoch = 5  # Unfreeze backbone after this epoch

# -------------------------------
# 4️⃣ Transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# 5️⃣ Load datasets
# -------------------------------
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Detected classes ({num_classes}): {class_names}")
print("Total train images:", len(train_dataset))
print("Total test images:", len(test_dataset))

# -------------------------------
# 6️⃣ Load MobileNetV2 model
# -------------------------------
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# Freeze backbone initially
for param in model.features.parameters():
    param.requires_grad = False

# -------------------------------
# 7️⃣ Loss and optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 8️⃣ Lists to store metrics for plotting
# -------------------------------
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# -------------------------------
# 9️⃣ Training loop with gradual unfreeze
# -------------------------------
for epoch in range(num_epochs):
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # After a few epochs, unfreeze and fine-tune:
    if epoch == 5:
        print("🔓 Unfreezing backbone for fine-tuning...")
        for param in model.features.parameters():
            param.requires_grad = True

        # Recreate optimizer with both groups (different learning rates)
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ])

    # -------- Training --------
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=120)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        train_bar.set_postfix({
            "Loss": f"{train_loss:.4f}",
            "Acc": f"{train_acc:.2f}%"
        })

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # -------- Validation --------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    val_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", ncols=120, leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_bar.set_postfix({
                "Val_Loss": f"{val_loss / val_total:.4f}",
                "Val_Acc": f"{100 * val_correct / val_total:.2f}%"
            })

    val_loss /= len(test_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # -------- Epoch summary --------
    print(f"Epoch [{epoch + 1}/{num_epochs}] Completed "
          f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# -------------------------------
# 10️⃣ Save the trained model
# -------------------------------
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
torch.save(model.state_dict(), save_model_path)
print(f"Model saved at: {save_model_path}")

# -------------------------------
# 11️⃣ Plot metrics
# -------------------------------
epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracies, 'bo-', label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_losses, 'bo-', label='Train Loss')
plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
