import cv2
import os

def create_squares_from_images(input_folder, output_folder):
    # Iterate through all files in the input folder
    for root, _, files in os.walk(input_folder):
        for filename in files:
            # Check if the file is an image
            if filename.endswith(('.png' )):
                # Construct the full input path
                img_path = os.path.join(root, filename)
                
                # Read the image
                img = cv2.imread(img_path)
                
                # Get the dimensions of the image
                height, width = img.shape[:2]
                
                # Calculate the number of 1200x1200 squares we can create
                num_squares = height // width
                
                for i in range(num_squares):
                    # Calculate the starting and ending y-coordinates for the square
                    start_y = i * width
                    end_y = start_y + width
                    
                    if end_y <= height:
                        # Extract the 1200x1200 square
                        square_img = img[start_y:end_y, :]
                        
                        # Construct the output path, preserving the folder structure
                        relative_path = os.path.relpath(root, input_folder)
                        output_subfolder = os.path.join(output_folder, relative_path)
                        os.makedirs(output_subfolder, exist_ok=True)
                        
                        # Construct the output filename
                        output_filename = f"{os.path.splitext(filename)[0]}_{i}.png"
                        output_path = os.path.join(output_subfolder, output_filename)
                        
                        # Save the square image
                        cv2.imwrite(output_path, square_img)
                        print(f"Saved square image: {output_path}")

# Example usage
input_folder = '../data/images'
output_folder = '../dataset/images'
create_squares_from_images(input_folder, output_folder)
