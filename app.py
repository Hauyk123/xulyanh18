import os
from flask import Flask, render_template, request, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
# Đảm bảo bạn có file segmentation.py trong cùng thư mục
from segmentation import segment_kmeans, segment_otsu, save_image_as_csv
import cv2

# Cấu hình Flask
app = Flask(__name__)
# Thêm một secret key, cần thiết để sử dụng flash messages
app.secret_key = 'hau-ptit-secret-key-for-project'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB

# Đảm bảo các thư mục tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # --- 1. Xử lý file upload ---
        if 'file' not in request.files:
            flash("LỖI: Không có file nào được gửi lên.", "error")
            return render_template('index.html')

        file = request.files['file']

        if file.filename == '':
            flash("LỖI: Bạn chưa chọn file ảnh nào.", "error")
            return render_template('index.html')

        if file:
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)
            base_filename, _ = os.path.splitext(filename)

            # --- 2. Lấy tùy chọn từ người dùng ---
            algorithms_to_run = request.form.getlist('algorithms')
            k_value = request.form.get('k_value', 3, type=int)

            if not algorithms_to_run:
                flash("Vui lòng chọn ít nhất một thuật toán để xử lý.", "warning")
                return render_template('index.html')

            # --- 3. Chạy các thuật toán được chọn ---
            results = {'original': original_path}

            if 'kmeans' in algorithms_to_run:
                kmeans_img = segment_kmeans(original_path, k=k_value)
                kmeans_filename = f"{base_filename}_kmeans_k{k_value}.png"
                kmeans_path = os.path.join(app.config['RESULT_FOLDER'], kmeans_filename)
                cv2.imwrite(kmeans_path, kmeans_img)
                results['kmeans'] = {'img_path': kmeans_path}
                results['k_value'] = k_value

            if 'otsu' in algorithms_to_run:
                otsu_img = segment_otsu(original_path)
                otsu_filename = f"{base_filename}_otsu.png"
                otsu_path = os.path.join(app.config['RESULT_FOLDER'], otsu_filename)
                cv2.imwrite(otsu_path, otsu_img)

                csv_filename = f"{base_filename}_otsu.csv"
                csv_path = os.path.join(app.config['RESULT_FOLDER'], csv_filename)
                save_image_as_csv(otsu_img, csv_path)

                results['otsu'] = {'img_path': otsu_path, 'csv_file': csv_filename}

            # --- 4. Trả kết quả về cho giao diện ---
            flash("Xử lý ảnh thành công!", "success")
            return render_template('index.html', results=results)

    # Nếu là GET request, chỉ hiển thị trang
    return render_template('index.html')


@app.route('/download/results/<filename>')
def download_result_file(filename):
    """Xử lý download file kết quả"""
    return send_from_directory(
        app.config['RESULT_FOLDER'],
        filename,
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)

