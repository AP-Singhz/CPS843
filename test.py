import torch
import cv2
from torchvision import models, transforms
import torch.nn as nn
from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np  

# Load the trained ResNet-50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  
model.load_state_dict(torch.load('resnet50_trained_model.pth', weights_only=True))  # Use weights_only=True
model.to(device)
model.eval()

# Define the transformations for the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 (required by ResNet)
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Open the webcam (usually 0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    barcodes = decode(frame)

    for barcode in barcodes:
        rect_points = barcode.polygon
        
        if len(rect_points) == 4:
            pts = np.array([rect_points], dtype=np.int32)  
        else:
            pts = np.array([rect_points], dtype=np.int32)
        
        if pts.size > 0:
            pts = cv2.convexHull(pts)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            x, y, w, h = cv2.boundingRect(pts)  
            barcode_image = frame[y:y+h, x:x+w] 

            barcode_pil = Image.fromarray(cv2.cvtColor(barcode_image, cv2.COLOR_BGR2RGB))

            # Apply transformations to the barcode image
            input_image = transform(barcode_pil).unsqueeze(0).to(device)

            # Make a prediction using the trained model
            with torch.no_grad():
                output = model(input_image)
                _, predicted = torch.max(output, 1)

            class_names = ['Intact', 'Defective'] 
            predicted_class = class_names[predicted.item()]

            # Display the predicted class on the frame
            cv2.putText(frame, f'Predicted: {predicted_class}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with barcode detection and prediction label
    cv2.imshow('Webcam Barcode Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
