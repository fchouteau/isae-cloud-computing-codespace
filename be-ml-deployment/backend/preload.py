"""
This script downloads the YOLO v11 models inside the dockerfile to avoid downloading them at runtime
"""
from ultralytics import YOLO

# Download and cache YOLO v11 models
_ = YOLO("yolo11n.pt")
_ = YOLO("yolo11s.pt")
_ = YOLO("yolo11m.pt")
