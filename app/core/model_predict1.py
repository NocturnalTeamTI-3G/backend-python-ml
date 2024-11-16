# import library 
import joblib
import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.image as mpimg


# Fungsi untuk memuat model
def load_model(model):
    return joblib.load(model)


# Fungsi untuk memuat satu gambar dari direktori
def load_img(img_path):
    img_path = Path(img_path)
    
    try:
        img = mpimg.imread(str(img_path))
    except (FileNotFoundError, OSError) as e:
        print(f"Error loading image: {e}")
        return None
    
    return img


# Function untuk melakukan grayscaling image
def grayscale_img(img_list):
    img_gray = []
    
    for img in img_list:
        img_gray.append((cv.cvtColor(img[0], cv.COLOR_RGB2GRAY), img[1]))
    
    return img_gray


# Function untuk melakukan ektraksi fitur dengan LBP
def extract_lbp_features(img_list, P=8, R=1):
    lbp_features = []
    
    for img in img_list:
        lbp = local_binary_pattern(img[0], P, R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-6)
        lbp_features.append((hist, img[1]))
    
    return lbp_features


# Function untuk melakukan standarisasi
def standardize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


# Fungsi utama untuk memuat gambar, melakukan grayscaling, ekstraksi fitur LBP, dan standarisasi
def preprocess_segment_and_extract_features(img_path):
    # Load gambar
    img = load_img(img_path)
    if img is None:
        return None

    # Konversi gambar ke format list (dengan label dummy karena tidak diperlukan pada prediksi)
    img_list = [(img, 0)]

    # Langkah 1: Konversi gambar ke grayscale
    img_gray = grayscale_img(img_list)

    # Langkah 2: Ekstraksi fitur LBP
    lbp_features = extract_lbp_features(img_gray)

    # Langkah 3: Standarisasi fitur
    features = np.array([f[0] for f in lbp_features])  # Hanya ambil histogram LBP
    standardized_features = standardize_features(features)

    return standardized_features[0]  # Kembalikan fitur dari gambar tunggal


# Fungsi untuk memprediksi jerawat pada gambar
def predict_jerawat(img_path, model_path):
    # Ekstraksi fitur dari gambar
    features = preprocess_segment_and_extract_features(img_path)
    if features is None:
        return "Gambar tidak valid atau gagal diproses."

    # Memuat model
    model = load_model(model_path)

    # Melakukan prediksi
    prediction = model.predict([features])

    return prediction[0]
