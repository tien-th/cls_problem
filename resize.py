import os
from PIL import Image
from tqdm import tqdm 

# Paths for input and output directories
input_root = "train_data"
output_root = "resized_train_data"

# Target resolution
target_size = (224, 224)

# Create the output directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Traverse the input folder
for subfolder in os.listdir(input_root):
    input_subfolder_path = os.path.join(input_root, subfolder)
    # output_subfolder_path = os.path.join(output_root, subfolder)
    
    if subfolder in ['bothcells', 'healthy']: 
        continue

    if os.path.isdir(input_subfolder_path):  # Check if it's a folder
        os.makedirs(output_root, exist_ok=True)  # Create the subfolder in the output directory

        for file_name in tqdm(os.listdir(input_subfolder_path)):
            if file_name.endswith(".png"):  # Check if it's a PNG image
                input_file_path = os.path.join(input_subfolder_path, file_name)
                output_file_path = os.path.join(output_root, file_name)

                try:
                    # Open and resize the image
                    with Image.open(input_file_path) as img:
                        resized_img = img.resize(target_size, Image.ANTIALIAS)
                        # Save to the output path
                        resized_img.save(output_file_path)
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")

print("Image resizing completed!")