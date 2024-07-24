import zipfile
import os
import shutil

def move_files(src_directory, dst_directory):
    # Ensure the destination directory exists
    os.makedirs(dst_directory, exist_ok=True)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(src_directory):
        for file in files:
            if file.lower().endswith('.img'):
                # Construct full file path
                file_path = os.path.join(root, file)
                # Move file to the destination directory
                shutil.move(file_path, dst_directory)

def list_immediate_subfolders(main_folder):
    subfolders = []
    for root, dirs, files in os.walk(main_folder):
        for dir_name in dirs:
            subfolders.append(os.path.join(root, dir_name))
        break  # Only process the top directory
    return subfolders

def extract_files(zip_file_path, extract_to_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get the list of all archived file names from the zip
        list_of_files = zip_ref.namelist()
        
        # Iterate over the file names
        for file_name in list_of_files:
            if file_name.endswith('.img'):
                # Extract the file
                zip_ref.extract(file_name, extract_to_folder)
                print(f'Extracted: {file_name}')


def find_and_extract(main_folder):
    # Walk through all subfolders of the main folder
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            # Check if the file is a ZIP file
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                extract_to_folder = root  # Extract to the same subfolder where the ZIP file is located
                extract_files(zip_file_path, extract_to_folder)

# Example usage
main_folder = '/media/soujatya/Seagate/OHRC/data/raw'
find_and_extract(main_folder)
all_directories = list_immediate_subfolders(main_folder)
for directory in all_directories:
    src_directory = directory
    dst_directory = directory.replace("/media/soujatya/Seagate/OHRC", ".")
    move_files(src_directory, dst_directory)
    print(f"All files have been moved to {dst_directory}")