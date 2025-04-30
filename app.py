import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

# Load YOLO model
@st.cache_resource
def load_model():
    net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights", "yolov4-tiny-custom.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    output_layers = [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(img, net, classes, output_layers):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        conf = round(confidences[i], 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf}", (x, y - 10), font, 0.5, (0, 255, 0), 2)
    return img

# Streamlit UI
st.title("YOLOv4-Tiny Object Detection")
st.markdown("Detect **cars** and **swimming pools** using your custom YOLOv4-tiny model")

option = st.radio("Select media type", ("Image", "Video"))

net, classes, output_layers = load_model()

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result_img = detect_objects(image, net, classes, output_layers)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_objects(frame, net, classes, output_layers)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        cap.release()
