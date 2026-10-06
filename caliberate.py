import os
import glob
import cv2
import numpy as np

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    # Crop central area where the container sits
    roi = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 1. Outer boundary irregularity (Dents alter the hull defect depth)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find outer contours of the container
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    # Intact bottles/cans: smooth perimeter (lower perimeter relative to area)
    # Crushed/crumpled bottles/cans: highly jagged perimeter
    jaggedness = (perimeter ** 2) / (area + 1e-5)
    
    # 2. Local gradient standard deviation (wrinkles vs smooth surface)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    wrinkle_var = laplacian.var()
    
    return round(jaggedness, 1), round(wrinkle_var, 1)

print("--- GOOD IMAGES ---")
for p in sorted(glob.glob("data/good/*.*")):
    print(f"{os.path.basename(p):15}: {extract_features(p)}")

print("\n--- BAD IMAGES ---")
for p in sorted(glob.glob("data/bad/*.*")):
    print(f"{os.path.basename(p):15}: {extract_features(p)}")