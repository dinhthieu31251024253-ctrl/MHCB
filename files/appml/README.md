# 🎵 Phân loại Nhạc AI vs Nhạc Thật

Ứng dụng web Streamlit cho phép tải lên file ZIP chứa các file audio, tự động
trích xuất đặc trưng âm thanh và dùng model Logistic Regression đã train để
dự đoán từng file là **Nhạc AI** hay **Nhạc do nhạc sĩ thật tạo**.

⚠️ **Lưu ý:** Model trong repo này là bản demo/thử nghiệm, được train trên
tập dữ liệu giới hạn. Kết quả dự đoán mang tính tham khảo, **không phải**
công cụ xác thực bản quyền hay pháp lý.

## Cách chạy local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc project

- `app.py` — ứng dụng Streamlit chính
- `model.pkl` — model Logistic Regression đã train (scikit-learn)
- `scaler.pkl` — StandardScaler dùng để chuẩn hóa đặc trưng trước khi predict
- `requirements.txt` — danh sách thư viện cần cài

## Đặc trưng âm thanh sử dụng

35 đặc trưng trích xuất bằng `librosa`: zero-crossing rate, RMS energy,
tempo, onset strength, spectral centroid/bandwidth/rolloff, 13 hệ số MFCC
(mean + std), và chroma.

## Deploy

Có thể deploy miễn phí qua [Streamlit Community Cloud](https://share.streamlit.io)
bằng cách kết nối trực tiếp với repo GitHub này.
