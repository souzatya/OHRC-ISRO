import cv2
import numpy as np
import os

def create_edge_mask(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    edges = cv2.GaussianBlur(edges, (21, 21), 0)
    
    # Create a mask based on the edges
    mask = np.zeros_like(image)
    
    # Set the non-edge regions to red
    mask[edges == 0] = [0, 0, 255]  # Red color in BGR format
    # Set the edge regions to black
    mask[edges != 0] = [0, 0, 0]
    
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
