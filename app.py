import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# Load YOLOv4-tiny custom model (ensure the weights and cfg file are in the same folder or specify the correct path)
def load_model():
    # Streamlit file uploader for weights and cfg
    weights_file = st.file_uploader("Upload YOLOv4-tiny Weights", type=['weights'])
    cfg_file = st.file_uploader("Upload YOLOv4-tiny Config", type=['cfg'])
    
    if weights_file is not None and cfg_file is not None:
        # Save the uploaded files to disk
        with open("yolov4-tiny-custom_best.weights", "wb") as f:
            f.write(weights_file.getbuffer())
        with open("yolov4-tiny-custom.cfg", "wb") as f:
            f.write(cfg_file.getbuffer())
        
        # Now load the model using the uploaded files
        net = cv2.dnn.readNet('yolov4-tiny-custom_best.weights', 'yolov4-tiny-custom.cfg')
        with open('obj.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers
    else:
        st.warning("Please upload the weights and config files")
        return None, None, None


# Run detection
def detect_objects(image, net, output_layers):
    # Get height, width, and channels of the input image
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Run forward pass to get the detections
    layer_outputs = net.forward(output_layers)
    
    # Initialize lists to hold detection info
    class_ids = []
    confidences = []
    boxes = []
    width_ratio = width / 416
    height_ratio = height / 416
    
    # Loop over the detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width_ratio)
                center_y = int(detection[1] * height_ratio)
                w = int(detection[2] * width_ratio)
                h = int(detection[3] * height_ratio)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

# Display image with bounding boxes and labels
def display_image(image, boxes, confidences, class_ids, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Convert BGR image to RGB for displaying
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# Streamlit app UI
def main():
    st.title("YOLOv4-tiny Object Detection")

    st.write("""
    Upload an image and let the model detect the car and swimming pool.
    """)

    # File uploader to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        # Read and process the uploaded image
        img = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Load model and perform object detection
        net, classes, output_layers = load_model()
        boxes, confidences, class_ids = detect_objects(image, net, output_layers)

        # Display the detected objects in the image
        display_image(image, boxes, confidences, class_ids, classes)

if __name__ == '__main__':
    main()
