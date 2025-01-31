import os
import urllib.request

# Function to download the OpenPose model files
def download_openpose_model():
  
    weights_file = "pose_iter_440000.caffemodel"
    base_url = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/models/pose/coco/"
    
    
    if not os.path.exists(weights_file):
        print("Downloading pose_iter_440000.caffemodel...")
        urllib.request.urlretrieve(base_url + weights_file, weights_file)

download_openpose_model()
