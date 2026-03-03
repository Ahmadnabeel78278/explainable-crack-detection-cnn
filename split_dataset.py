import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Source folder containing 'positive' and 'negative'
source_dir = 'data'
# Target folder where split data will be stored
target_dir = 'data_splitted'

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

# Create target directories
for split in ['train', 'val', 'test']:
    for class_name in ['positive', 'negative']:
        os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)

# Process each class
for class_name in ['positive', 'negative']:
    class_path = os.path.join(source_dir, class_name)
    images = os.listdir(class_path)
    
    # Split into train+val and test
    train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
    # Split train+val into train and val
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    
    # Copy files
    for img in train:
        shutil.copy(os.path.join(class_path, img), os.path.join(target_dir, 'train', class_name, img))
    for img in val:
        shutil.copy(os.path.join(class_path, img), os.path.join(target_dir, 'val', class_name, img))
    for img in test:
        shutil.copy(os.path.join(class_path, img), os.path.join(target_dir, 'test', class_name, img))

print("Dataset splitting completed!")
print(f"Train: {len(os.listdir(os.path.join(target_dir, 'train', 'positive')))} positive, {len(os.listdir(os.path.join(target_dir, 'train', 'negative')))} negative")
print(f"Val: {len(os.listdir(os.path.join(target_dir, 'val', 'positive')))} positive, {len(os.listdir(os.path.join(target_dir, 'val', 'negative')))} negative")
print(f"Test: {len(os.listdir(os.path.join(target_dir, 'test', 'positive')))} positive, {len(os.listdir(os.path.join(target_dir, 'test', 'negative')))} negative")