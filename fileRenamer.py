
import os

def rename_images(folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    # List all files in the folder
    files = os.listdir(folder_path)
    images = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]  # Filter image files
    images.sort()  # Optional: Sort to rename in a specific order
    
    for i, filename in enumerate(images, start=1):
        # Get file extension
        file_ext = os.path.splitext(filename)[1]
        # Construct new file name
        new_name = f"image{i}{file_ext}"
        # Rename file
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")
        
# Path to your folder with images
folder_path = r"C:\Users\samri\Downloads\barcode_bb\barcode_bb\non_defective"
rename_images(folder_path)
