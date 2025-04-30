import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
@st.cache_resource
def load_model():
    net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights", "yolov4-tiny-custom.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    return net, classes, output_layers

# Function to process frames
def process_frame(frame, net, output_layers):
    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detections
    height, width, channels = frame.shape
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Perform non-maxima suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Streamlit UI
st.title("Swimming Pool and Car Detection")

# Load the model
net, classes, output_layers = load_model()

# Select between image and video input
input_type = st.radio("Choose Input Type", ("Image", "Video"))

if input_type == "Image":
    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Process the image
        processed_img = process_frame(img, net, output_layers)
        
        # Show the result
        st.image(processed_img, channels="BGR", caption="Processed Image", use_column_width=True)

elif input_type == "Video":
    # Upload video
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        video_file = cv2.imdecode(np.frombuffer(video_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Process video frame by frame
        st.video(uploaded_video)
