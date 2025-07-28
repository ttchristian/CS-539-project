import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from timm import create_model

# ==== Convert to RGB ====
class ConvertToRGB:
    def __call__(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
# ==== MultiTask ViT ====
class MultiTaskViT(nn.Module):
    def __init__(self, num_fruit_classes = 14, num_health_classes = 2):
        super(MultiTaskViT, self).__init__()

        self.backbone = create_model(
            'vit_base_patch16_224',
            pretrained = True,
            num_classes = 0
        )
        hidden_dim = self.backbone.num_features

        self.classifier_fruit = nn.Linear(hidden_dim, num_fruit_classes)
        self.classifier_health = nn.Linear(hidden_dim, num_health_classes)

    def forward(self, x):
        features = self.backbone(x)
        out_fruit = self.classifier_fruit(features)
        out_health = self.classifier_health(features)
        return out_fruit, out_health
    
# ==== Dataset Wrapper ====
class MultiTaskDataset(Dataset):
    def __init__(self, imagefolder_dataset, idx_to_fruit_health):
        self.base = imagefolder_dataset
        self.idx_map = idx_to_fruit_health

    def __getitem__(self, index):
        image, class_idx = self.base[index]
        fruit_id, health_id = self.idx_map[class_idx]
        return image, torch.tensor(fruit_id), torch.tensor(health_id)

    def __len__(self):
        return len(self.base)
    
# ==== Hyperparameter ==== 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes_fruit = 14
num_classes_health = 2
batch_size = 128
lr = 1e-4
max_epochs = 30
early_stop_patience = 5

transform = transforms.Compose([
    ConvertToRGB(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_folder = ImageFolder("/home/chris/projects/fruit-disease-detect/fruit_disease_dataset/train", transform=transform)
val_folder = ImageFolder("/home/chris/projects/fruit-disease-detect/fruit_disease_dataset/val", transform=transform)

class_to_idx = train_folder.class_to_idx
idx_to_fruit_health = {}
fruit_name_to_id = {}
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

train_dataset = MultiTaskDataset(train_folder, idx_to_fruit_health)
val_dataset = MultiTaskDataset(val_folder, idx_to_fruit_health)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

model = MultiTaskViT(num_classes_fruit, num_classes_health).to(device)

criterion_fruit = nn.CrossEntropyLoss()
criterion_health = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ==== Evaluate ====
def evaluate(loader, epoch=None):
    model.eval()
    correct_fruit = 0
    correct_health = 0
    total = 0
    with torch.no_grad():
        for images, label_fruit, label_health in loader:
            images = images.to(device)
            label_fruit = label_fruit.to(device)
            label_health = label_health.to(device)
            out_fruit, out_health = model(images)
            _, pred_fruit = out_fruit.max(1)
            _, pred_health = out_health.max(1)
            correct_fruit += pred_fruit.eq(label_fruit).sum().item()
            correct_health += pred_health.eq(label_health).sum().item()
            total += images.size(0)
    return correct_fruit / total, correct_health / total

# ==== Logging ====
def log_line(message, log_path="training_log.txt"):
    with open(log_path, "a") as f:
        f.write(message + "\n")

# ==== Training ====
def train():
    best_val_score = 0.0
    trigger_times = 0
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    log_path = "training_log.txt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_fruit = 0
        correct_health = 0
        total = 0

        for images, label_fruit, label_health in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            label_fruit = label_fruit.to(device)
            label_health = label_health.to(device)
            optimizer.zero_grad()
            out_fruit, out_health = model(images)
            loss_fruit = criterion_fruit(out_fruit, label_fruit)
            loss_health = criterion_health(out_health, label_health)
            total_loss = loss_fruit + loss_health
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            _, pred_fruit = out_fruit.max(1)
            _, pred_health = out_health.max(1)
            correct_fruit += pred_fruit.eq(label_fruit).sum().item()
            correct_health += pred_health.eq(label_health).sum().item()
            total += images.size(0)

        train_acc_fruit = correct_fruit / total
        train_acc_health = correct_health / total
        val_acc_fruit, val_acc_health = evaluate(val_loader, epoch=epoch)
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        train_accuracies.append((train_acc_fruit, train_acc_health))
        val_accuracies.append((val_acc_fruit, val_acc_health))
        train_losses.append(avg_loss)

        log_msg = (
            f"[Epoch {epoch}] Loss: {avg_loss:.4f}, "
            f"Train Fruit Acc: {train_acc_fruit:.4f}, Train Health Acc: {train_acc_health:.4f}, "
            f"Val Fruit Acc: {val_acc_fruit:.4f}, Val Health Acc: {val_acc_health:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        print(log_msg)
        log_line(log_msg, log_path)

        val_score = (val_acc_fruit + val_acc_health) / 2
        scheduler.step(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pth")
            log_line(f"✅ Best model updated (Val score = {val_score:.4f})", log_path)
        else:
            trigger_times += 1
            log_line(f"⚠️ No improvement for {trigger_times} epoch(s).", log_path)

        if trigger_times >= early_stop_patience:
            stop_msg = f"⏹️ Early stopping triggered at epoch {epoch}"
            print(stop_msg)
            log_line(stop_msg, log_path)
            break

    # ==== Draw Curves ====
    epochs_range = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, [x[0] for x in train_accuracies], label="Train Fruit Acc")
    plt.plot(epochs_range, [x[0] for x in val_accuracies], label="Val Fruit Acc")
    plt.title("Fruit Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, [x[1] for x in train_accuracies], label="Train Health Acc")
    plt.plot(epochs_range, [x[1] for x in val_accuracies], label="Val Health Acc")
    plt.title("Health Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_losses, label="Train Loss", color='orange')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves_vit_multitask.png")
    plt.show()

if __name__ == "__main__":
    train()