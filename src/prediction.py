import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import csv  # ✅ Added for saving results

# -------------------------------
# 1️⃣ Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2️⃣ Dataset paths
# -------------------------------
train_dir = "datasets/train"
test_dir = "datasets/test"

# -------------------------------
# 3️⃣ Transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

# -------------------------------
# 4️⃣ Load dataset to get class names
# -------------------------------
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
class_names = train_dataset.classes
print("Classes detected:", class_names)

# -------------------------------
# 5️⃣ Load model
# -------------------------------
num_classes = len(class_names)
model = models.mobilenet_v2(weights=None, num_classes=num_classes)  # Same as trained
model.load_state_dict(torch.load(r"C:\Users\mrans\Desktop\PlantDiseaseAI\models\mobilenetv2_model.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# 6️⃣ Function to predict an image
# -------------------------------
def predict_image(image_path, model, transform, class_names):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_idx].item() * 100  # ✅ Added: Probability %

    return class_names[pred_idx], pred_prob  # ✅ Return both class & prob


# -------------------------------
# 7️⃣ Predict for all test images
# -------------------------------
results = []  # ✅ Store predictions to save later

for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)
            pred_class, pred_prob = predict_image(img_path, model, transform, class_names)

            print(f"Image: {file} --> Predicted Disease: {pred_class} ({pred_prob:.2f}%)")

            # ✅ Save result to list
            results.append({
                "Image": file,
                "Predicted Disease": pred_class,
                "Probability (%)": f"{pred_prob:.2f}"
            })

# -------------------------------
# 8️⃣ Save predictions to CSV file
# -------------------------------
output_csv = "predictions1.csv"
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Image", "Predicted Disease", "Probability (%)"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ All predictions saved to: {output_csv}")
