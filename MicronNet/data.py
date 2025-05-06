from __future__ import print_function
import zipfile
import os
import shutil # Import shutil for moving files, potentially more robust

import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.Scale((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

def initialize_data(folder):
    # Use os.path.join for robust path creation
    train_zip = os.path.join(folder, 'train_images.zip')
    test_zip = os.path.join(folder, 'test_images.zip')

    # It's good practice to uncomment this check if the files are required
    # if not os.path.exists(train_zip) or not os.path.exists(test_zip):
    #     raise(RuntimeError(f"Could not find {train_zip} or {test_zip}"
    #           + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2017/data '))

    train_folder = os.path.join(folder, 'train_images')
    if not os.path.isdir(train_folder):
        # Check if zip file exists before trying to extract
        if os.path.exists(train_zip):
            print(f'{train_folder} not found, extracting {train_zip}')
            try:
                with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                    zip_ref.extractall(folder)
            except Exception as e:
                print(f"Error extracting {train_zip}: {e}")
                return # Exit if extraction fails
        else:
             print(f"Error: {train_folder} not found and {train_zip} does not exist.")
             return # Exit if source is missing

    test_folder = os.path.join(folder, 'test_images')
    if not os.path.isdir(test_folder):
        if os.path.exists(test_zip):
            print(f'{test_folder} not found, extracting {test_zip}')
            try:
                with zipfile.ZipFile(test_zip, 'r') as zip_ref:
                    zip_ref.extractall(folder)
            except Exception as e:
                print(f"Error extracting {test_zip}: {e}")
                # Decide if you want to return here or continue without test data
        else:
            print(f"Warning: {test_folder} not found and {test_zip} does not exist.")
            # Decide if you want to return or continue

    val_folder = os.path.join(folder, 'val_images')
    if not os.path.isdir(val_folder):
        print(f'{val_folder} not found, making a validation set')
        try:
            os.makedirs(val_folder, exist_ok=True) # Use makedirs with exist_ok=True
            for item_name in os.listdir(train_folder):
                source_item_path = os.path.join(train_folder, item_name)
                # Check if it's a directory AND starts with '000'
                if os.path.isdir(source_item_path) and item_name.startswith('000'):
                    target_dir_path = os.path.join(val_folder, item_name)
                    os.makedirs(target_dir_path, exist_ok=True) # Create corresponding val dir
                    
                    for f in os.listdir(source_item_path):
                        if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                            source_file_path = os.path.join(source_item_path, f)
                            target_file_path = os.path.join(target_dir_path, f)
                            # Check if file exists before moving
                            if os.path.isfile(source_file_path):
                                print(f"Moving {source_file_path} to {target_file_path}")
                                try:
                                    # Use shutil.move for potentially better cross-device handling
                                    shutil.move(source_file_path, target_file_path)
                                except Exception as move_e:
                                    print(f"Error moving {source_file_path}: {move_e}")
                            else:
                                print(f"Warning: Expected file {source_file_path} not found.")

        except OSError as e:
            print(f"Error creating validation directory structure: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during validation set creation: {e}")

    # Optional: Add return statements if needed, e.g., return train_folder, val_folder, test_folder
    print("Data initialization process finished.")

# Example usage (replace 'path/to/your/data' with the actual path)
initialize_data('d:\\justi\\Code\\FYP\\MicronNet-master') # Or your actual data path
# initialize_data('path/to/your/data')