import cv2

# Initialize the camera (0 for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a frame
ret, frame = cap.read()

if ret:
    # Save the image
    cv2.imwrite("captured_image.jpg", frame)
    print("Image saved as captured_image.jpg")

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
