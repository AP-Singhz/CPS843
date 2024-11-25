

# This is the full code but requires the model file

# import cv2
# import numpy as np
# import torch
# from torchvision import transforms
# from pyzbar.pyzbar import decode

# # Load the pre-trained PyTorch model
# class BarcodeCNN(torch.nn.Module):
#     def __init__(self):
#         super(BarcodeCNN, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = torch.nn.Linear(32 * 32 * 32, 128)
#         self.fc2 = torch.nn.Linear(128, 2)  # Binary classification: readable vs. flawed

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, kernel_size=2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, kernel_size=2)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Initialize model and load weights
# model = BarcodeCNN()
# model.load_state_dict(torch.load("barcode_model.pth"))   # pytorch model file name here
# model.eval()  # Set the model to evaluation mode

# # Preprocessing for PyTorch model
# transform = transforms.Compose([
#     transforms.Grayscale(),  # Convert to grayscale
#     transforms.Resize((128, 128)),  # Resize to match model input
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize((0.5,), (0.5,))  # Normalize
# ])

# # Start the camera
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)  # Set width
# cap.set(4, 640)  # Set height

# print("Press 'q' to quit.")
# try:
#     while True:
#         success, img = cap.read()
#         if not success:
#             print("Failed to capture image.")
#             break

#         # Process each barcode in the frame
#         for barcode in decode(img):
#             try:
#                 # Decode barcode data
#                 data = barcode.data.decode('utf-8')
#                 print(f"Decoded Barcode Data: {data}")
                
#                 # Get coordinates of the barcode
#                 x, y, w, h = barcode.rect
#                 print(f"Barcode Coordinates: x={x}, y={y}, w={w}, h={h}")

#                 # Extract the region of interest (ROI)
#                 roi = img[y:y+h, x:x+w]
                
#                 # Preprocess the ROI for PyTorch
#                 if roi.size > 0:
#                     roi_tensor = transform(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).unsqueeze(0)
#                     prediction = model(roi_tensor)
#                     _, label = torch.max(prediction, 1)
                    
#                     # Determine if the barcode is readable or flawed
#                     result_text = "Readable" if label.item() == 0 else "Flawed"
#                     print(f"Classification Result: {result_text}")

#                     # Draw results on the frame
#                     color = (0, 255, 0) if label.item() == 0 else (0, 0, 255)
#                     cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
#                     cv2.putText(img, result_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#             except Exception as e:
#                 print(f"Error processing barcode: {e}")

#         # Display the video feed with barcode annotations
#         cv2.imshow("Barcode Scanner", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     cap.release()
#     cv2.destroyAllWindows()

# this code below is just for testing that it gets the tensor data from the camera
import cv2
import numpy as np
from torchvision import transforms
from pyzbar.pyzbar import decode
from PIL import Image  # For converting NumPy arrays to PIL Images
import torch

# Preprocessing for PyTorch model
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((128, 128)),  # Resize to match model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Optional: Print full tensors
torch.set_printoptions(threshold=10_000)  # Adjust threshold to avoid truncation in large tensors

# Start the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 640)  # Set height

print("Press 'q' to quit.")
try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        # Process each barcode in the frame
        for barcode in decode(img):
            try:
                # Decode barcode data
                data = barcode.data.decode('utf-8')
                print(f"Decoded Barcode Data: {data}")
                
                # Get coordinates of the barcode
                x, y, w, h = barcode.rect
                print(f"Barcode Coordinates: x={x}, y={y}, w={w}, h={h}")

                # Extract the ROI (Region of Interest)
                roi = img[y:y+h, x:x+w]
                if roi.size > 0:
                    # Convert the ROI from NumPy array to PIL Image
                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    
                    # Preprocess ROI
                    roi_tensor = transform(roi_pil).unsqueeze(0)
                    print(f"Tensor Shape: {roi_tensor.shape}")
                    print(f"Tensor Data:\n{roi_tensor}")
            except Exception as e:
                print(f"Error processing barcode: {e}")

        # Display the video feed
        cv2.imshow("Barcode Scanner", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
