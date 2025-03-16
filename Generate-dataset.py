import os
import random
import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision.models.detection as models
from pycocotools.coco import COCO

# Check for GPU availability
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
# Load pre-trained Mask R-CNN model
model = models.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# Define output directories
forged_output_dir = "./forged_images"
mask_output_dir = "./mask_images"
os.makedirs(forged_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

# Load COCO dataset
dataset_path = "./coco_dataset"
image_dir = os.path.join(dataset_path, "train2017")
coco_path = os.path.join(dataset_path, "annotations/instances_train2017.json")
coco = COCO(coco_path)

def load_image(image_id):
    """Load an image from COCO dataset."""
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    return image, img_info['file_name']

def detect_objects(image):
    """Detect objects in an image using Mask R-CNN."""
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    masks = predictions[0]['masks'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    boxes = predictions[0]['boxes'].cpu().numpy()

    # Lower confidence threshold to detect smaller objects
    min_confidence = 0.1
    valid_indices = [i for i, score in enumerate(scores) if score > min_confidence]

    if not valid_indices:
        return None  # No valid objects detected

    detected_objects = {
        'masks': [masks[i][0] for i in valid_indices],
        'labels': [labels[i] for i in valid_indices],
        'scores': [scores[i] for i in valid_indices],
        'boxes': [boxes[i] for i in valid_indices]
    }

    return detected_objects

def adjust_brightness(image, factor):
    """Adjust brightness of an image using a given factor."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)  # Adjust brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def blend_objects(image, obj_crop, mask_crop, new_x, new_y):
    """Blend pasted object into the background using Poisson or Alpha blending."""
    try:
        center = (new_x + obj_crop.shape[1] // 2, new_y + obj_crop.shape[0] // 2)
        blended_image = cv2.seamlessClone(obj_crop, image, (mask_crop * 255).astype(np.uint8), center, cv2.NORMAL_CLONE)
        return blended_image, "BL"
    except:
        # Fallback to simple alpha blending
        alpha = 0.6  # Adjust transparency
        region = image[new_y:new_y+obj_crop.shape[0], new_x:new_x+obj_crop.shape[1]]
        blended_region = cv2.addWeighted(region, alpha, obj_crop, 1 - alpha, 0)
        image[new_y:new_y+obj_crop.shape[0], new_x:new_x+obj_crop.shape[1]] = blended_region
        return image, "BL"

def create_forgery_and_mask(image, detections, image_id, category):
    """Create a forged image with random object scaling, rotation, brightness, blending, and mask."""
    if not detections:
        return image, None, ""

    idx = np.random.randint(0, len(detections['masks']))  # Choose a random detected object
    mask = (detections['masks'][idx] > 0.5).astype(np.uint8)

    obj = cv2.bitwise_and(image, image, mask=mask)

    # Find object bounding box
    y_indices, x_indices = np.where(mask > 0)
    y_min, y_max, x_min, x_max = y_indices.min(), y_indices.max(), x_indices.min(), x_indices.max()

    obj_crop = obj[y_min:y_max+1, x_min:x_max+1]
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
    obj_h, obj_w = obj_crop.shape[:2]

    # Apply random scaling (0.5x to 1.5x)
    scale_factor = random.choice([1.0] * 8 + [random.uniform(0.5, 1.2) for _ in range(2)])
    new_w, new_h = int(obj_w * scale_factor), int(obj_h * scale_factor)
    scaled = scale_factor != 1.0

    resized_obj_crop = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask_crop = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Apply random rotation (mostly no rotation, sometimes ±30, ±45, ±60, ±90 degrees)
    rotation_choices = [None] * 16 + [30, 45, 60, 90, -30, -45, -60, -90]
    rotation_angle = random.choice(rotation_choices)
    rotated = rotation_angle is not None
    if rotated:
        center = (new_w // 2, new_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        resized_obj_crop = cv2.warpAffine(resized_obj_crop, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
        resized_mask_crop = cv2.warpAffine(resized_mask_crop, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

    # Apply random brightness change (rarely applied)
    brightness_suffix = ""
    if random.random() < 0.2:
        brightness_factor = random.uniform(0.7, 1.3)
        resized_obj_crop = adjust_brightness(resized_obj_crop, brightness_factor)
        brightness_suffix = "B"

    h, w, _ = image.shape
    new_x = random.randint(0, max(1, w - new_w))
    new_y = random.randint(0, max(1, h - new_h))

    # Blending (20% chance)
    blend_suffix = ""
    if random.random() < 0.2:
        image, blend_suffix = blend_objects(image, resized_obj_crop, resized_mask_crop, new_x, new_y)
    else:
        image[new_y:new_y+new_h, new_x:new_x+new_w][resized_mask_crop > 0] = resized_obj_crop[resized_mask_crop > 0]

    # Create mask image
    mask_image = np.zeros_like(image, dtype=np.uint8)
    mask_image[mask > 0] = [0, 255, 0]  # Green for original
    mask_image[new_y:new_y+new_h, new_x:new_x+new_w][resized_mask_crop > 0] = [0, 0, 255]  # Red for pasted

    suffix = brightness_suffix + blend_suffix
    if rotated:
        suffix += "R"
    if scaled:
        suffix += "S"

    return image, mask_image, suffix

# Generate dataset
num_images = 50000
image_ids = random.sample(coco.getImgIds(), num_images)

for img_id in image_ids:
    image, file_name = load_image(img_id)
    if image is None:
        continue
    
    detections = detect_objects(image)

    if detections:
        category = detections['labels'][0]
        forged_image, mask_image, suffix = create_forgery_and_mask(image, detections, img_id, category)
        
        if forged_image is not None:
            cv2.imwrite(os.path.join(forged_output_dir, f"TP_{img_id}_{category}{suffix}.jpg"), forged_image)
            cv2.imwrite(os.path.join(mask_output_dir, f"Mask_{img_id}_{category}{suffix}.jpg"), mask_image)

print("Forgery dataset generation complete.")
