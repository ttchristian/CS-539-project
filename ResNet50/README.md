# Fruit Disease Detection using ResNet50 (Multi-Task Learning)

This project uses a ResNet50-based deep learning model to classify **fruit types** and **health status (healthy or rotten)** from images. It implements **multi-task learning**, allowing simultaneous classification of two targets.

---

## 📁 Project Structure

```bash
.
├── data_split.py               # Download dataset and split into train/val/test (8:1:1)
├── data.py                     # Fix double file extension in image files
├── ResNet50-fruit-disease.py  # Train the multi-task ResNet50 model
├── test_acc.py                 # Evaluate model accuracy on test set
├── multi_task_gradcam.py      # Generate Grad-CAM visualizations for misclassified images
├── model/                      # Folder to save trained models
├── results/                    # Folder to save Grad-CAM outputs
└── README.md
