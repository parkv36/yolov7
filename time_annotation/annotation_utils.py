import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def extract_features(img, hsv):
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    hue = np.mean(hsv[:, :, 0])
    saturation = np.mean(hsv[:, :, 1])
    value = np.mean(hsv[:, :, 2])
    
    red_channel = np.mean(img[:, :, 2])
    blue_channel = np.mean(img[:, :, 0])
    color_temp = blue_channel / (red_channel + 1e-5)
    
    return [brightness, hue, saturation, value, color_temp]

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # img = cv2.resize(img, (224, 224))
    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # # Apply histogram equalization to the V (brightness) channel
    # hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # Optional (better local contrast): use CLAHE instead
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    return img, hsv

def preprocess_ir_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read IR image: {img_path}")

    img = cv2.resize(img, (224, 224))

    # Optionally enhance local contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img  # grayscale

def extract_ir_features(img):
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)

    # Edge density
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / img.size

    # Laplacian variance (texture/focus)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

    # Entropy from intensity histogram
    hist = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
    hist /= hist.sum() + 1e-5
    entropy = -np.sum(hist * np.log2(hist + 1e-5))

    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)
    lbp_hist = lbp_hist.tolist()

    return [mean_intensity, std_intensity, edge_density, laplacian_var, entropy] + lbp_hist

def extract_rgb_highlevel_features(img, hsv):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    h, w = gray.shape
    mid_region_mean = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
    top_mean = np.mean(gray[:h//2, :])
    bottom_mean = np.mean(gray[h//2:, :])

    dark_frac = np.sum(gray < 50) / gray.size
    bright_frac = np.sum(gray > 220) / gray.size

    saturation = np.mean(hsv[:, :, 1])
    red = np.mean(img[:, :, 2])
    blue = np.mean(img[:, :, 0])
    color_temp = blue / (red + 1e-5)

    return [
        mean_intensity, std_intensity, mid_region_mean,
        top_mean, bottom_mean, dark_frac, bright_frac,
        saturation, color_temp
    ]

def extract_ir_highlevel_features(gray):
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    h, w = gray.shape
    mid_region_mean = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
    top_mean = np.mean(gray[:h//2, :])
    bottom_mean = np.mean(gray[h//2:, :])

    dark_frac = np.sum(gray < 50) / gray.size
    bright_frac = np.sum(gray > 220) / gray.size

    return [
        mean_intensity, std_intensity, mid_region_mean,
        top_mean, bottom_mean, dark_frac, bright_frac
    ]
