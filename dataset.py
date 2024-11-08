import os
import cv2
from torch.utils.data import Dataset
from PIL import Image

class barcode_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        for label, category in enumerate(["non_defective", "defective"]):
            category_dir = os.path.join(root_dir, category)
            print("Looking for Directory: ", category_dir) # Debugging line to see if correct path is found
            
            if not os.path.exists(category_dir):
                raise FileNotFoundError(f"Directory not found: {category_dir}")
            
            for file_name in os.listdir(category_dir):
                file_path = os.path.join(category_dir, file_name)
                self.data.append((file_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.GaussianBlur(image, (5,5), 0)
        edges = cv2.Canny(image, threshold1=50, threshold2=150)
        return edges
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = self.preprocess_image(image_path)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

