import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

# ============ define multitask model ============
class MultiTaskResNet(nn.Module):
    def __init__(self, num_fruit_classes=14, num_health_classes=2):
        super(MultiTaskResNet, self).__init__()
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 去掉 avgpool & fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier_fruit = nn.Linear(2048, num_fruit_classes)
        self.classifier_health = nn.Linear(2048, num_health_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.avgpool(features)
        flat = self.flatten(pooled)
        return self.classifier_fruit(flat), self.classifier_health(flat)

# ============ hyperparameter ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/home/chris/projects/fruit-disease-detect/fruit_disease_dataset/test"
output_dir = "misclassified_output"
os.makedirs(output_dir, exist_ok=True)


fruit_name_to_idx = {}
health_name_to_idx = {}
for idx, name in enumerate(sorted(os.listdir(dataset_path))):
    try:
        fruit_part, health_part = name.split("_", 1)
    except ValueError:
        continue  
    if fruit_part not in fruit_name_to_idx:
        fruit_name_to_idx[fruit_part] = len(fruit_name_to_idx)
    if health_part not in health_name_to_idx:
        health_name_to_idx[health_part] = len(health_name_to_idx)


class ConvertToRGB:
    def __call__(self, image):
        return image.convert("RGB")

transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


test_dataset = ImageFolder(dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = MultiTaskResNet(num_fruit_classes=len(fruit_name_to_idx),
                        num_health_classes=len(health_name_to_idx))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device).eval()


target_layer = model.backbone[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

for i, (img_tensor, true_label) in enumerate(tqdm(test_loader)):
    img_tensor = img_tensor.to(device)
    true_classname = test_dataset.classes[true_label.item()]
    
    try:
        fruit_name, health_name = true_classname.split("_", 1)
        true_fruit = fruit_name_to_idx[fruit_name]
        true_health = health_name_to_idx[health_name]
    except ValueError:
        continue

    with torch.no_grad():
        out_fruit, out_health = model(img_tensor)
        pred_fruit = out_fruit.argmax(dim=1).item()
        pred_health = out_health.argmax(dim=1).item()

    if pred_fruit != true_fruit or pred_health != true_health:
        # Grad-CAM for fruit branch
        grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_fruit)])[0]


        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_rgb = (img_np * 255).astype(np.uint8)
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)


        pred_fruit_name = list(fruit_name_to_idx.keys())[pred_fruit]
        pred_health_name = list(health_name_to_idx.keys())[pred_health]
        save_name = f"wrong_{fruit_name}_{health_name}_pred_{pred_fruit_name}_{pred_health_name}_{i}.jpg"
        save_path = os.path.join(output_dir, save_name)
        side_by_side = np.concatenate((img_rgb, cam_image), axis=1)
        cv2.imwrite(save_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        print(f"❌ Saved misclassified: {save_path}")

print("all mistake grad-cam images are generated")

