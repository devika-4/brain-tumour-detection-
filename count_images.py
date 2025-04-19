import os

print("Script started...")  # Debugging print

# Path to the dataset
dataset_path = r"C:\BrainTumorDataset\Training"

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset path '{dataset_path}' does not exist.")
else:
    # Get class-wise image count
    for category in ["glioma", "meningioma", "pituitary", "notumor"]:
        folder_path = os.path.join(dataset_path, category)
        
        if not os.path.exists(folder_path):
            print(f"Warning: '{category}' folder not found.")
            continue  # Skip this folder
        
        num_images = len(os.listdir(folder_path))
        print(f"{category}: {num_images} images")

