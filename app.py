import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
def load_model():
    net = cv2.dnn.readNet('yolov4-tiny-custom_best.weights', 'yolov4-tiny-custom.cfg')
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers

# Perform detection
def detect_objects(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(416, 416), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

# Draw detection labels
def draw_labels(img, boxes, confidences, class_ids, classes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Streamlit UI
def main():
    st.title("YOLOv4-Tiny Object Detection (Cars & Pools)")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image_np, caption='Uploaded Image', use_column_width=True)

        net, classes, output_layers = load_model()
        boxes, confidences, class_ids = detect_objects(image_np, net, output_layers)
        result_img = draw_labels(image_np.copy(), boxes, confidences, class_ids, classes)

        st.image(result_img, caption='Detection Result', use_column_width=True)

if __name__ == "__main__":
    main()
