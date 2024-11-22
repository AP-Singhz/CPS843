import cv2
import numpy as np
from pyzbar.pyzbar import decode

# img = cv2.imread('BC1.png') # reads the barcode and gets barcode data from file
# code = decode(img)

cap = cv2.VideoCapture(0)  # 0 indicates first camera connected to computer
cap.set(3, 640)  # sets the width; 3 is the property ID (identify properties), 640 is the pixels
cap.set(4, 640)  # sets the height; 4 is the property ID (identify properties), 640 is the pixels
x = 20
try:
    print("\nPress 'q' to quit the application.\n")  # User instruction

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image\n")  # Safety check in case the camera feed fails
            break
            
        for barcode in decode(img):  # multiple barcodes
            # print(barcode.data)
            # print(barcode.rect)
            try:
                myData = barcode.data.decode('utf-8')  # removes the string and gives actual data
                print(myData)
                coords = np.array([barcode.polygon], np.int32)  # holds coordinates of vertices of polygon surrounding barcode
                coords = coords.reshape((-1, 1, 2))  # reshapes array to have -1 rows, 1 second dimension, and 2 for x and y coordinates
                cv2.polylines(img, [coords], True, (255, 0, 0), 11)  # draws polygon on image using points specified by coords; 11 represents thickness

                coords2 = barcode.rect  # rectangle around the barcode
                cv2.putText(img, myData, (coords2[0], coords2[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)  # puts barcode data on screen, (coords2[0], coords2[1]) is where text is drawn, 2 is font size, (0, 0, 255) is color (red), 2 is text thickness
            except Exception as e:
                print(f"Error decoding barcode: {e}\n")  # Error handling for unexpected decoding issues

        cv2.imshow('Result', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
            print("\nExiting program... \n")
            break

except Exception as e:
    print(f"An error occurred: {e}\n")  # General error handling
finally:
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("\nCamera released, and all windows closed.\n")