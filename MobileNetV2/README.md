# Fruit Scan AI with MobileNetV2

---
## Environment Setup

Create and activate a virtual environment, then install the required libraries:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision pillow
```

---

## Dataset Preparation

Organize your dataset into subfolders by class. Then run `reduce_dataset.py` to create a smaller dataset for faster training:

```bash
python reduce_dataset.py
```

The script will generate a new folder with the reduced dataset.

---

## Model Training

Run `main.py` to train the MobileNetV2 model:

```bash
python main.py
```

---

## 4. Image Prediction

Use `predict.py` to classify an image with the trained model:

```bash
python predict.py
```

The program will output the predicted class and confidence score.
