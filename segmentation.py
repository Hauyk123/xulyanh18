import cv2
import numpy as np
import os




def segment_kmeans(image_path, k=2):
    """
    Phân đoạn ảnh sử dụng K-means (thuần numpy).
    """

    # 1. Đọc ảnh bằng OpenCV (sẽ ra BGR)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # 2. Chuyển sang RGB (vì logic K-means thuần của bạn xử lý RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # === Logic K-means thuần (từ xulianh-main/src/image_processor.py) ===
    h, w, c = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    # 1. Khởi tạo ngẫu nhiên tâm cụm
    np.random.seed(42)
    centers = pixels[np.random.choice(len(pixels), k, replace=False)]

    max_iter = 100
    tol = 1e-4

    for _ in range(max_iter):
        # 2. Tính khoảng cách Euclidean
        distances = np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Cập nhật tâm cụm
        new_centers = np.array([
            pixels[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])

        # 4. Kiểm tra hội tụ
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers

    # 5. Tạo ảnh kết quả (ảnh này đang là RGB)
    segmented_pixels = centers[labels].astype(np.uint8)
    kmeans_result_rgb = segmented_pixels.reshape((h, w, c))
    # === Hết logic K-means thuần ===

    # 6. Chuyển kết quả về BGR để app.py (dùng cv2.imwrite) lưu cho đúng
    kmeans_result_bgr = cv2.cvtColor(kmeans_result_rgb, cv2.COLOR_RGB2BGR)

    return kmeans_result_bgr


def segment_otsu(image_path):
    """
    Phân đoạn ảnh sử dụng thuật toán Otsu (thuần numpy).
    """

    # 1. Đọc ảnh
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # 2. Chuyển sang RGB (giống logic file image_processor.py)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # === Logic Otsu thuần (từ xulianh-main/src/image_processor.py) ===
    # 3. Chuyển sang grayscale
    gray = np.dot(image_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    # 4. Tính histogram
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    prob = hist / total

    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), prob)
    sumB, wB = 0, 0

    # 5. Lặp qua các ngưỡng
    for t in range(256):
        wB += prob[t]
        if wB == 0:
            continue
        wF = 1 - wB
        if wF == 0:
            break
        sumB += t * prob[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = t

    # 6. Áp ngưỡng
    binary_image = (gray > threshold).astype(np.uint8) * 255
    # === Hết logic Otsu thuần ===

    return binary_image


def save_image_as_csv(image_array, output_csv_path):
    """
    Lưu mảng ảnh (kết quả Otsu) thành file CSV.
    (Giữ nguyên hàm này)
    """
    # Lưu ma trận 2D (0 và 255) vào file CSV
    np.savetxt(output_csv_path, image_array, delimiter=',', fmt='%d')