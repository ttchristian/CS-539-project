import os
import shutil
import random
from pathlib import Path

random.seed(42)  


origin_dataset_dir = "/home/chris/.cache/kagglehub/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/versions/1/Fruit And Vegetable Diseases Dataset"

target_base_dir = "fruit_disease_dataset"


train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1


for split in ['train', 'val', 'test']:
    for class_name in os.listdir(origin_dataset_dir):
        os.makedirs(os.path.join(target_base_dir, split, class_name), exist_ok=True)


for class_name in os.listdir(origin_dataset_dir):
    img_dir = Path(origin_dataset_dir) / class_name
    imgs = list(img_dir.glob("*"))
    random.shuffle(imgs)
    
    n_total = len(imgs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    for i, img_path in enumerate(imgs):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        dst = Path(target_base_dir) / split / class_name / img_path.name
        if os.path.isdir(img_path):
            continue

        shutil.copy(img_path, dst)

print("dataset all set ")
