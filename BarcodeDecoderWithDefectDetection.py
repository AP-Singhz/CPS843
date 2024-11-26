import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Barcode Detector
detector = cv2.barcode.BarcodeDetector()

# Define transformations (without ToPILImage as the image is already a PIL image)
transform = transforms.Compose([
    transforms.Resize((128, 128)),     # Resize image to 128x128 pixels
    transforms.Grayscale(),            # Convert to grayscale
    transforms.ToTensor(),             # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
])

# Define the SimpleCNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust based on input dimensions
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load your pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load(r'C:\Users\samri\OneDrive\Desktop\CPS843\CPS843\model.pth'))  # Load the saved model weights
model.eval()  # Set the model to evaluation mode

# OpenCV to capture video
cap = cv2.VideoCapture(0)  # Use webcam (0 is the default webcam)

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break  # If frame not captured, break the loop
    
    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect barcode in the frame
    retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(gray_frame)
    
    if retval:
        # Barcode detected
        for i in range(len(decoded_info)):
            print(f"Detected barcode: {decoded_info[i]}")
            points_array = np.array(points[i], dtype=int)
            cv2.polylines(frame, [points_array], True, (0, 255, 0), 3)  # Draw polygon around the barcode

        # Use barcode presence for classification
        barcode_detected = 1  # If barcode detected
    else:
        # No barcode detected
        barcode_detected = 0
        cv2.putText(frame, "No Barcode Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Preprocess the frame for the model (same as dataset transformation)
    pil_image = Image.fromarray(frame)  # Convert to PIL Image
    tensor_image = transform(pil_image).unsqueeze(0)  # Add batch dimension

    # Pass the frame through the model
    with torch.no_grad():  # No need to track gradients for inference
        output = model(tensor_image)  # Forward pass

    # Apply sigmoid to get prediction (output close to 1 means defect, 0 means no defect)
    prediction = torch.sigmoid(output)
    print(f"Model Prediction: {'Defective' if prediction.item() > 0.5 else 'Non-Defective'}")

    # Display the frame
    cv2.imshow('Webcam Barcode and Defect Detection', frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
