# import library 
import joblib
import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern, hog
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

# Function untuk melakukan resizing image
def resize_img(img_list, size=(200, 200)):
    img_resized = []
    for img in img_list:
        img_resized.append((cv.resize(img[0], size, interpolation=cv.INTER_AREA), img[1]))
    return img_resized

# Function untuk melakukan ektraksi fitur dengan LBP
def extract_lbp_features(img_list, P=8, R=1):
    features = []

    for img, label in img_list:
        if len(img.shape) != 3:
            print(f"Gambar tidak valid: {img.shape}")

        lbp_features = []
        hog_features = []

        # Ekstraksi fitur dari setiap channel
        for channel in range(3):  # Iterasi melalui R, G, B
            single_channel = img[..., channel]

            # LBP untuk channel saat ini
            lbp = local_binary_pattern(single_channel, P, R, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-6)  # Normalisasi histogram
            lbp_features.append(hist)

            # HOG untuk channel saat ini
            hog_feat, _ = hog(
                single_channel,  # Asumsi gambar grayscale per channel
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                visualize=True
            )
            hog_features.append(hog_feat)

        # Gabungkan semua fitur dari ketiga channel
        combined_lbp = np.hstack(lbp_features)
        combined_hog = np.hstack(hog_features)
        combined_features = np.hstack((combined_lbp, combined_hog))

        # Tambahkan ke daftar fitur
        features.append((combined_features, label))

    return features


# Function untuk melakukan standarisasi
def standardize_features(features, scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler.transform(features)


# Fungsi utama untuk memuat gambar, melakukan grayscaling, ekstraksi fitur LBP, dan standarisasi
def preprocess_segment_and_extract_features(img_path, scaler_path):
    # Load gambar
    img = load_img(img_path)
    if img is None:
        return None

    # Konversi gambar ke format list (dengan label dummy karena tidak diperlukan pada prediksi)
    img_list = [(img, 0)]

    img_resize = resize_img(img_list, (200, 200))
    
    # Langkah 1: Konversi gambar ke grayscale
    # img_gray = grayscale_img(img_resize)

    # Langkah 2: Ekstraksi fitur LBP
    lbp_features = extract_lbp_features(img_resize)

    # Langkah 3: Standarisasi fitur
    features = np.array([f[0] for f in lbp_features])  # Hanya ambil histogram LBP
    standardized_features = standardize_features(features, scaler_path)

    return standardized_features[0]  # Kembalikan fitur dari gambar tunggal


# Fungsi untuk memprediksi jerawat pada gambar
def predict_jerawat(img_path, model_path, scaler_path):
    # Ekstraksi fitur dari gambar
    features = preprocess_segment_and_extract_features(img_path, scaler_path)
    if features is None:
        return "Gambar tidak valid atau gagal diproses."

    # Memuat model
    model = load_model(model_path)

    # Melakukan prediksi
    prediction = model.predict([features])

    return prediction[0]
