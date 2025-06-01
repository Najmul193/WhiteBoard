import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def detect_red_ink(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_binary = np.where(red_mask > 0, 0, 1).astype(np.uint8)  # writing = 0
    return red_binary

def detect_writing_with_manual_threshold(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, "Error loading image."

    image = cv2.resize(image, (800, 600))
    h, w, _ = image.shape
    margin_y = int(0.10 * h)
    margin_x = int(0.10 * w)
    center_crop = image[margin_y:h - margin_y, margin_x:w - margin_x]

    gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    ch, cw = gray.shape
    mid_x = cw // 2
    left_half_gray = gray[:, :mid_x]
    right_half_gray = gray[:, mid_x:]

    left_half_color = center_crop[:, :mid_x]
    right_half_color = center_crop[:, mid_x:]

    crop_margin_l = int(0.10 * left_half_gray.shape[1])
    crop_margin_r = int(0.10 * right_half_gray.shape[1])
    left_cropped_gray = left_half_gray[:, crop_margin_l:-crop_margin_l]
    right_cropped_gray = right_half_gray[:, crop_margin_r:-crop_margin_r]

    left_cropped_color = left_half_color[:, crop_margin_l:-crop_margin_l]
    right_cropped_color = right_half_color[:, crop_margin_r:-crop_margin_r]

    left_norm = left_cropped_gray / 255.0
    right_norm = right_cropped_gray / 255.0

    left_binary_gray = np.where(left_norm < 0.4, 0, 1).astype(np.uint8)
    right_binary_gray = np.where(right_norm < 0.4, 0, 1).astype(np.uint8)

    left_binary_red = detect_red_ink(left_cropped_color)
    right_binary_red = detect_red_ink(right_cropped_color)

    left_binary = np.minimum(left_binary_gray, left_binary_red)
    right_binary = np.minimum(right_binary_gray, right_binary_red)

    total_left_pixels = left_binary.size
    total_right_pixels = right_binary.size

    left_black_pixels = np.sum(left_binary == 0)
    right_black_pixels = np.sum(right_binary == 0)

    left_black_pct = (left_black_pixels / total_left_pixels) * 100
    right_black_pct = (right_black_pixels / total_right_pixels) * 100

    threshold_pct = 1
    left_result = 1 if left_black_pct >= threshold_pct else 0
    right_result = 1 if right_black_pct >= threshold_pct else 0

    def save_image(img, fname):
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        cv2.imwrite(path, img)
        return path

    center_crop_path = save_image(center_crop, "center_crop.jpg")
    left_binary_path = save_image(left_binary * 255, "left_binary.jpg")
    right_binary_path = save_image(right_binary * 255, "right_binary.jpg")

    return {
        "left_result": left_result,
        "right_result": right_result,
        "left_black_pct": left_black_pct,
        "right_black_pct": right_black_pct,
        "center_crop": center_crop_path,
        "left_binary": left_binary_path,
        "right_binary": right_binary_path
    }, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        results, error = detect_writing_with_manual_threshold(filepath)
        if error:
            return f"<h3>{error}</h3>"

        for key in ['center_crop', 'left_binary', 'right_binary']:
            results[key] = url_for('uploaded_file', filename=os.path.basename(results[key]))

        return render_template('result.html', results=results)

    return render_template('index.html')

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)