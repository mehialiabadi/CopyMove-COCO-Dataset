# CopyMove-COCO-Dataset

## 📌 Overview
# CopyMove-COCO-Dataset

## 📌 Overview
This project generates a **Copy-Move Forgery Dataset** using the **COCO dataset**. It detects objects in images, transforms and copies them to a new location within the same image, and creates corresponding ground truth masks.

## 🚀 How It Works
1. **Load COCO Dataset**: Reads images from the COCO dataset.
2. **Object Detection**: Uses a pre-trained Mask R-CNN model to detect objects.
3. **Transformation**:
   - Scaling (randomly between 0.5x to 1.2x)
   - Rotation (random angles: ±30°, ±45°, ±60°, ±90°)
   - Brightness adjustments (random, 20% chance)
4. **Copy-Move Forgery**:
   - Pastes the object at a new random location.
   - Uses blending techniques (Poisson or Alpha blending) for realism.
5. **Ground Truth Mask Generation**:
   - **Green (0, 255, 0)**: Original object region.
   - **Red (0, 0, 255)**: Forged/copied object region.
## 📂 Output Structure
CopyMove-COCO-Dataset/
│── forged_images/       # Forged images
│   ├── TP_123456_3B.jpg   # Forged image (TP: Tampered, Category 3, Brightness Adjusted)
│   ├── TP_789012_5BL.jpg  # Forged image (Category 5, Blended)
│── mask_images/         # Corresponding masks
│   ├── Mask_123456_3B.jpg  # Mask (Green: Original, Red: Forged)
│   ├── Mask_789012_5BL.jpg
│── generate_dataset.py   # Main dataset generation script
│── requirements.txt      # Dependencies
│── README.md             # Documentation

File Naming Convention
TP_123456_3B.jpg → TP_ (Tampered image) from COCO image 123456, Category 3, Brightness adjusted.
Mask_123456_3B.jpg → Corresponding mask for the image above.
- **Suffixes in filenames**:
  - **B**: Brightness adjustment applied
  - **BL**: Blending applied
  - **R**: Rotated
  - **S**: Scaled



## 🏗 Installation
To run the dataset generation script, follow these steps:

```bash
git clone https://github.com/mehialiabadi/CopyMove-COCO-Dataset.git
cd CopyMove-COCO-Dataset
pip install -r requirements.txt

##📜 License
This project is open-source under the MIT License.

🙌 Acknowledgments
COCO Dataset
