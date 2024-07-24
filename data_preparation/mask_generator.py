import cv2
import torch
import numpy as np
import os
import supervision as sv
from collections import Counter

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def binarize_mask(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    pixels_tuple = [tuple(pixel) for pixel in pixels]
    color_counts = Counter(pixels_tuple)
    most_common_color = color_counts.most_common(1)[0][0]

    # Create masks
    mask_most_common = np.all(image_rgb == most_common_color, axis=-1)
    mask_other = ~mask_most_common

    # Replace colors
    image_rgb[mask_most_common] = [0, 0, 0]
    image_rgb[mask_other] = [255, 255, 255]

    # Convert the image back to BGR format
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr

def create_mask(image_path):
    image = cv2.imread(image_path)
    image_black = np.zeros((1200, 1200, 3), dtype=np.uint8)

    sam = sam_model_registry["vit_h"](checkpoint="./sam_ckpt/sam_vit_h_4b8939.pth").to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=masks)
    annotated_image = mask_annotator.annotate(scene = image_black.copy(), detections=detections)
    
    mask = binarize_mask(annotated_image)

    return mask

def process_dataset(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.png'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                mask = create_edge_mask(input_path)
                if mask is not None:
                    cv2.imwrite(output_path, mask)
                    print(f"Saved edge mask for {input_path} to {output_path}")

# Example usage
input_folder = '../data_processed/images'
output_folder = '../data_processed/masks'
process_dataset(input_folder, output_folder)
