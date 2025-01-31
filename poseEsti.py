import cv2
import numpy as np

# Load the model
proto_file = "pose_deploy_linevec.prototxt"
weights_file = "pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

# Load an image
image_path = "test_image.jpg"  # Replace with your image file
frame = cv2.imread(image_path)

# Get image dimensions
h, w, _ = frame.shape
in_width = 368
in_height = 368

# Pre-process the image for the model
inp_blob = cv2.dnn.blobFromImage(
    frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False
)
net.setInput(inp_blob)

# Perform pose estimation
output = net.forward()

# Threshold for confidence
threshold = 0.1

# Draw points on the image
for i in range(output.shape[1]):
    prob_map = output[0, i, :, :]
    _, prob, _, point = cv2.minMaxLoc(prob_map)

    x = int((point[0] * w) / output.shape[3])
    y = int((point[1] * h) / output.shape[2])

    if prob > threshold:
        cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1)

# Show the image
cv2.imshow("Pose Estimation", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
