import cv2
import numpy as np
from edge_impulse_linux.runner import ImpulseRunner

def main():
    model_path = 'modelfile.eim'  # Replace with your model's filename
    image_path = 'apple.jpg'      # Replace with your image file

    # Initialize the Edge Impulse model runner
    runner = ImpulseRunner(model_path)
    try:
        model_info = runner.init()
        labels = model_info['model_parameters']['labels']

        # Load and preprocess the image
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        desired_h = model_info['model_parameters']['image_input_height']
        desired_w = model_info['model_parameters']['image_input_width']
        img_resized = cv2.resize(img, (desired_w, desired_h))

        # Perform inference
        features = np.array(img_resized).flatten().tolist()
        res = runner.classify(features)

        # Process results
        if 'bounding_boxes' in res['result'].keys():
            for bb in res['result']['bounding_boxes']:
                label = bb['label']
                if label.lower() == 'Apple':
                    print("Hi, I am apple")
                    # Optional: Draw bounding box
                    x1 = int(bb['x'] * w / desired_w)
                    y1 = int(bb['y'] * h / desired_h)
                    x2 = int((bb['x'] + bb['width']) * w / desired_w)
                    y2 = int((bb['y'] + bb['height']) * h / desired_h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the image with bounding boxes
            cv2.imshow('Detected Objects', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No objects detected.")

    finally:
        runner.stop()

if __name__ == '__main__':
    main()
