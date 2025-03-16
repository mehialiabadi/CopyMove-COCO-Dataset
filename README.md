# CopyMove-COCO-Dataset

## 📌 Overview
This project generates a **Copy-Move Forgery Dataset** using the **COCO dataset**. The dataset consists of images where objects are detected, transformed, pasted at different locations, and blended realistically to simulate copy-move forgery. Various augmentation techniques such as brightness and color intensity modifications are applied to ensure diversity.

## 📂 Dataset Generation Pipeline
The dataset creation follows these steps:
1. **Object Detection:** Identify objects in COCO images using a pre-trained model (Mask R-CNN).
2. **Transformation:** Apply geometric transformations (scaling, rotation, flipping) to detected objects.
3. **Copy-Move:** Paste the transformed object at a different location within the same image.
4. **Blending:** Smoothly blend the copied object with the background.
5. **Augmentation:** Adjust brightness, contrast, and color intensity.
6. **Mask Generation:** Create ground truth masks for the manipulated regions.
7. **Dataset Storage:** Save images along with their corresponding masks.

## 📌 Features
- **Large-Scale Dataset:** Generates up to 100K forged images.
- **Diverse Transformations:** Objects undergo various transformations to create realistic forgeries.
- **Automated Pipeline:** Fully automated dataset generation using COCO.
- **Ground Truth Masks:** Masks available for supervised learning in forgery detection.

## 🏗 Installation
To run the dataset generation script, follow these steps:

```bash
git clone https://github.com/mehialiabadi/CopyMove-COCO-Dataset.git
cd CopyMove-COCO-Dataset
pip install -r requirements.txt
