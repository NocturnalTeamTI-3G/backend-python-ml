#import library
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
# proses terbaik untuk saat ini
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2 as cv


def load_model(model):
    model = joblib.load(model)
    return model


# Fungsi untuk memuat satu gambar dari direktori
def load_img(img_path):
    # Mengonversi img_path menjadi Path object jika belum
    img_path = Path(img_path)
    
    try:
        img = mpimg.imread(str(img_path))
    except (FileNotFoundError, OSError) as e:
        print(f"Error loading image: {e}")
        return None
    
    return img

# Function to resize images and encoding labels

def resize_img_and_encode_label(img):
    resized_img = cv.resize(img, (224, 224))

    return resized_img

# Function to pre process images
def preprocess(img):
    std_img = resize_img_and_encode_label(img)
    return std_img


# Fungsi untuk mengubah gambar ke grayscale
def to_grayscale(img):
    return rgb2gray(img)

# Fungsi untuk segmentasi jerawat dengan thresholding
def segment_jerawat(img_gray):
    # Otsu's threshold untuk segmentasi otomatis
    thresh = threshold_otsu(img_gray)
    binary_mask = img_gray > thresh  # Thresholding jerawat, area jerawat akan bernilai True

    # Menghilangkan objek kecil yang mungkin bukan jerawat
    cleaned_mask = remove_small_objects(binary_mask, min_size=50)

    # Mengaplikasikan mask ke gambar grayscale untuk fokus pada area jerawat
    segmented_img = img_gray * cleaned_mask

    return segmented_img

# Fungsi untuk mengekstraksi fitur GLCM
def extract_glcm_features(segmented_img):
    # Mendefinisikan jarak dan sudut yang akan digunakan untuk GLCM
    distances = [1]  # Jarak pasangan piksel (1 pixel)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Sudut pasangan piksel: 0, 45, 90, dan 135 derajat

    # Menghitung GLCM
    glcm = graycomatrix(segmented_img.astype(np.uint8),
                        distances=distances,
                        angles=angles,
                        levels=256,
                        symmetric=True,
                        normed=True)

    # Ekstraksi fitur dari GLCM
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'ASM').mean()  # ASM juga dikenal sebagai Energy
    correlation = graycoprops(glcm, 'correlation').mean()

    # Kembalikan hasil ekstraksi fitur sebagai list numerik
    return [contrast, dissimilarity, homogeneity, energy, correlation]

# StandardScaler untuk normalisasi fitur?

# Fungsi utama untuk melakukan ekstraksi fitur dari list gambar
def preprocess_segment_and_extract_features(img):
    # Langkah 1: Konversi gambar ke grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    # Langkah 2: Segmentasi (contoh menggunakan thresholding sederhana)
    _, segmented_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)

    # Langkah 3: Ekstraksi fitur GLCM
    features = extract_glcm_features(segmented_img)

    # Gabungkan fitur dan label menjadi satu tuple
    return features

# Fungsi untuk memprediksi jerawat pada gambar
def predict_jerawat(img_path, model):
    # Memuat gambar
    img = load_img(img_path)
    if img is None:
        return None

    # Pra-pemrosesan gambar
    preprocessed_img = preprocess(img)

    # Ekstraksi fitur dari gambar
    features = preprocess_segment_and_extract_features(preprocessed_img)

    # Memuat model KNN
    model = load_model(model)

    # Melakukan prediksi
    prediction = model.predict([features])

    return prediction[0]