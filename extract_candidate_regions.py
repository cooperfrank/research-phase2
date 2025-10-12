import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def extract_candidate_regions(img_a_path, img_b_path, threshold=0.3, min_area=50):
    """
    Identify candidate regions of change between two screenshots using SSIM.

    Compares two images, computes a structural similarity (SSIM) difference map,
    and extracts bounding boxes around areas with significant visual changes.

    Parameters:
        img_a_path (str): Path to the first image (e.g., new screenshot).
        img_b_path (str): Path to the second image (e.g., reference screenshot).
        threshold (float, optional): SSIM difference threshold (0-1) for highlighting changes. Defaults to 0.3.
        min_area (int, optional): Minimum pixel area for a region to be considered significant. Defaults to 50.

    Returns:
        List[np.ndarray]: List of image patches (as NumPy arrays) corresponding to regions of change.
    """
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray_a, gray_b, full=True)
    print("SSIM:", score)

    # Prepare diff for visualization
    diff = (1 - diff) * 255
    diff = diff.astype("uint8")

    _, thresh = cv2.threshold(diff, int(threshold * 255), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patches = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        patches.append(img_a[y:y+h, x:x+w])

    return patches

if __name__ == "__main__":
    pass