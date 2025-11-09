import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights='IMAGENET1K_V1').features[:5]
model.eval()

image_path = r"C:\Users\mrans\Desktop\PlantDiseaseAI\datasets\test\test\PotatoHealthy1.JPG"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

img = Image.open(image_path).convert("RGB")
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    feature_maps = model(img_t)

for i in range(min(6, feature_maps.shape[1])):
    plt.subplot(2,3,i+1)
    plt.imshow(feature_maps[0][i].cpu(), cmap='viridis')
    plt.axis('off')

plt.show()
