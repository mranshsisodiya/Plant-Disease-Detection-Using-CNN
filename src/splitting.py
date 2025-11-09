from sklearn.model_selection import train_test_split
import os, shutil, random
from tqdm import tqdm

data_dir = "datasets/merged_dataset"
train_dir = "datasets/train"
test_dir = "datasets/test"

for dir_ in [train_dir, test_dir]:
    os.makedirs(dir_, exist_ok=True)

classes = os.listdir(data_dir)

for cls in tqdm(classes, desc="📂 Processing classes", unit="class"):
    imgs = os.listdir(os.path.join(data_dir, cls))
    train, test = train_test_split(imgs, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Copy training images with progress bar
    for img in tqdm(train, desc=f"🟢 Copying train → {cls}", unit="img", leave=False):
        shutil.copy(os.path.join(data_dir, cls, img), os.path.join(train_dir, cls, img))

    # Copy testing images with progress bar
    for img in tqdm(test, desc=f"🔵 Copying test → {cls}", unit="img", leave=False):
        shutil.copy(os.path.join(data_dir, cls, img), os.path.join(test_dir, cls, img))
