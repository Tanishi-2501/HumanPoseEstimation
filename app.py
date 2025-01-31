import cv2
import numpy as np
import streamlit as st
import os

# OpenPose model files (Update paths)
PROTO_FILE = "pose_deploy_linevec.prototxt"
WEIGHTS_FILE = "pose_iter_440000.caffemodel"

# Load OpenPose model
net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)

# Streamlit UI
st.title("Human Pose Estimation with OpenPose")
st.write("Upload an image to detect human pose keypoints.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    h, w, _ = image.shape
    in_width = 368
    in_height = 368

    # Preprocess image
    inp_blob = cv2.dnn.blobFromImage(
        image, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False
    )
    net.setInput(inp_blob)
    output = net.forward()

    # Define keypoint connections
    keypoint_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),  # Face
        (8, 9), (9, 10),  # Mouth
        (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # Left Arm
        (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),  # Right Arm
        (23, 24), (24, 25), (25, 26), (26, 27),  # Left Leg
        (28, 29), (29, 30), (30, 31), (31, 32)  # Right Leg
    ]

    # Draw keypoints
    for i in range(output.shape[1]):
        prob_map = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(prob_map)
        x = int((point[0] * w) / output.shape[3])
        y = int((point[1] * h) / output.shape[2])

        if prob > 0.1:  # Confidence threshold
            cv2.circle(image, (x, y), 5, (0, 255, 255), thickness=-1)

    # Draw skeleton connections
    for pair in keypoint_pairs:
        partA, partB = pair
        if partA < len(output[0]) and partB < len(output[0]):
            prob_mapA = output[0, partA, :, :]
            prob_mapB = output[0, partB, :, :]
            _, probA, _, pointA = cv2.minMaxLoc(prob_mapA)
            _, probB, _, pointB = cv2.minMaxLoc(prob_mapB)

            xA = int((pointA[0] * w) / output.shape[3])
            yA = int((pointA[1] * h) / output.shape[2])
            xB = int((pointB[0] * w) / output.shape[3])
            yB = int((pointB[1] * h) / output.shape[2])

            if probA > 0.1 and probB > 0.1:
                cv2.line(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Convert BGR to RGB for Streamlit display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show output
    st.image(image, caption="Pose Detection Output", use_column_width=True)
