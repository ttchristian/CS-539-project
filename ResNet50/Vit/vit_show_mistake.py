import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from timm import create_model
import os
import shutil

class MultiTaskViT(torch.nn.Module):
    def __init__(self, num_fruit_classes=14, num_health_classes=2):
        super(MultiTaskViT, self).__init__()
        self.backbone = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        hidden_dim = self.backbone.num_features
        self.classifier_fruit = torch.nn.Linear(hidden_dim, num_fruit_classes)
        self.classifier_health = torch.nn.Linear(hidden_dim, num_health_classes)

    def forward(self, x):
        features = self.backbone(x)
        out_fruit = self.classifier_fruit(features)
        out_health = self.classifier_health(features)
        return out_fruit, out_health


class ConvertToRGB:
    def __call__(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_classes_fruit = 14
num_classes_health = 2


test_folder = ImageFolder("/home/chris/projects/fruit-disease-detect/fruit_disease_dataset/test")
class_to_idx = test_folder.class_to_idx

fruit_name_to_id = {}
idx_to_fruit_health = {}
next_fruit_id = 0

for class_name, idx in class_to_idx.items():
    parts = class_name.split('_')
    fruit_part = '_'.join(parts[:-1])
    health_part = parts[-1]
    if fruit_part not in fruit_name_to_id:
        fruit_name_to_id[fruit_part] = next_fruit_id
        next_fruit_id += 1
    fruit_id = fruit_name_to_id[fruit_part]
    health_id = 0 if health_part.lower() == "healthy" else 1
    idx_to_fruit_health[idx] = (fruit_id, health_id)

id_to_fruit = {v: k for k, v in fruit_name_to_id.items()}
id_to_health = {0: "healthy", 1: "rotten"}


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, idx_map, transform=None):
        self.base = base_dataset
        self.idx_map = idx_map
        self.transform = transform

    def __getitem__(self, index):
        image, class_idx = self.base[index]
        path = self.base.samples[index][0]
        if self.transform:
            image = self.transform(image)
        fruit_id, health_id = self.idx_map[class_idx]
        return image, torch.tensor(fruit_id), torch.tensor(health_id), path

    def __len__(self):
        return len(self.base)


transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


test_dataset = MultiTaskDataset(test_folder, idx_to_fruit_health, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = MultiTaskViT(num_fruit_classes=num_classes_fruit, num_health_classes=num_classes_health)
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(device)
model.eval()


error_dir = "misclassified_examples_vit"
os.makedirs(error_dir, exist_ok=True)


correct_fruit = 0
correct_health = 0
total = 0

with torch.no_grad():
    for images, fruit_labels, health_labels, paths in test_loader:
        images = images.to(device)
        fruit_labels = fruit_labels.to(device)
        health_labels = health_labels.to(device)

        out_fruit, out_health = model(images)
        _, pred_fruit = out_fruit.max(1)
        _, pred_health = out_health.max(1)

        if pred_fruit.item() != fruit_labels.item() or pred_health.item() != health_labels.item():

            filename = os.path.basename(paths[0])
            true_fruit = id_to_fruit[fruit_labels.item()]
            true_health = id_to_health[health_labels.item()]
            pred_fruit_name = id_to_fruit[pred_fruit.item()]
            pred_health_name = id_to_health[pred_health.item()]
            save_name = f"wrong_{true_fruit}_{true_health}_pred_{pred_fruit_name}_{pred_health_name}_{filename}"
            save_path = os.path.join(error_dir, save_name)
            shutil.copy(paths[0], save_path)

        correct_fruit += pred_fruit.eq(fruit_labels).sum().item()
        correct_health += pred_health.eq(health_labels).sum().item()
        total += images.size(0)

acc_fruit = correct_fruit / total
acc_health = correct_health / total

print(f"🎯 Final Test Accuracy - Fruit: {acc_fruit:.4f}, Health: {acc_health:.4f}")
print(f"❌ Misclassified images saved to: {error_dir}")
