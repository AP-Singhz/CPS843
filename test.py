import cv2
import numpy as np  # Import numpy

# Initialize the barcode detector
detector = cv2.barcode.BarcodeDetector()

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Adjust camera ID if necessary

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect barcodes in the frame
    success, barcode_boxes = detector.detect(frame)

    # Check if any barcodes are detected
    if success and barcode_boxes is not None and len(barcode_boxes) > 0:
        print(f"Detected {len(barcode_boxes)} barcodes.")  # Debug print
        # Loop through each detected barcode
        for box in barcode_boxes:
            # Extract the box coordinates
            x, y, w, h = box  # x, y are top-left corner, w and h are width and height

            # Create the points array for decoding
            points = np.array([
                [x, y],              # Top-left
                [x + w, y],          # Top-right
                [x + w, y + h],      # Bottom-right
                [x, y + h]           # Bottom-left
            ], dtype=np.float32)

            # Print points for debugging
            print(f"Points for decoding: {points}")

            # Decode the barcode using the bounding box points
            decoded_info = detector.decode(frame, points)

            # Draw a rectangle around the detected barcode
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if the decoding was successful
            if decoded_info is not None:
                for info in decoded_info:
                    cv2.putText(frame, f'Detected: {info}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print("Decoding failed.")

    else:
        print("No barcodes detected.")

    # Display the frame
    cv2.imshow('Barcode Detection', frame)

    # Set delay to slow down refresh rate (e.g., 100 milliseconds)
    if cv2.waitKey(5000) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
