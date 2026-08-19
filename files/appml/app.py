import streamlit as st
import zipfile
import os
import tempfile
import joblib
import numpy as np
import librosa

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
st.set_page_config(
    page_title="Phân loại Nhạc AI vs Nhạc Thật",
    page_icon="🎵",
    layout="wide",
)

DURATION = 20         # Số giây audio đọc để trích xuất đặc trưng
OFFSET = 0.0          # ĐÃ SỬA: về 0.0 để khớp với lúc train (Colab cắt từ đầu bài, không offset)
SAMPLE_RATE = 22050   # Tần số lấy mẫu
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")

# Bật/tắt in log debug ra terminal Streamlit (không hiện trên giao diện web)
DEBUG_MODE = True

# Thứ tự đặc trưng PHẢI khớp chính xác với lúc scaler được train
FEATURE_ORDER = [
    "zcr_mean", "rms_mean", "tempo", "onset_mean", "onset_std",
    "spec_centroid_mean", "spec_bandwidth_mean", "spec_rolloff_mean",
    "mfcc1_mean", "mfcc1_std", "mfcc2_mean", "mfcc2_std",
    "mfcc3_mean", "mfcc3_std", "mfcc4_mean", "mfcc4_std",
    "mfcc5_mean", "mfcc5_std", "mfcc6_mean", "mfcc6_std",
    "mfcc7_mean", "mfcc7_std", "mfcc8_mean", "mfcc8_std",
    "mfcc9_mean", "mfcc9_std", "mfcc10_mean", "mfcc10_std",
    "mfcc11_mean", "mfcc11_std", "mfcc12_mean", "mfcc12_std",
    "mfcc13_mean", "mfcc13_std", "chroma_mean",
]

# 0 = Nhạc thật, 1 = Nhạc AI
LABEL_MAP = {
    0: "🎤 Nhạc do nhạc sĩ thật tạo",
    1: "🤖 Nhạc AI",
}
LABEL_COLOR = {
    0: "#27ae60",  # Xanh lá
    1: "#e74c3c",  # Đỏ
}


# ============================================================
# 1. TẢI MÔ HÌNH (cache để không load lại mỗi lần tương tác)
# ============================================================
@st.cache_resource
def load_model_and_scaler():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ============================================================
# HÀM PHỤ: TÍNH TEMPO — ĐÃ SỬA để khớp CHÍNH XÁC với lúc train
# Lúc train (Colab) dùng: librosa.beat.beat_track(y=y, sr=sr)
# nên ở đây phải dùng đúng cùng 1 hàm, không dùng librosa.feature.rhythm.tempo
# (2 hàm khác nhau có thể cho ra giá trị tempo khác nhau cho cùng 1 đoạn audio,
#  làm lệch phân bố dữ liệu đưa vào scaler/model)
# ============================================================
def get_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


# ============================================================
# 2 & 3. TRÍCH XUẤT 35 ĐẶC TRƯNG ÂM THANH
# ============================================================
def extract_features(file_path: str) -> np.ndarray:
    # offset=0.0 (mặc định) để khớp đúng lúc train: cắt đoạn ĐẦU bài, không dịch chuyển
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, offset=OFFSET, duration=DURATION, mono=True)

    if y is None or len(y) == 0:
        raise ValueError("File audio rỗng hoặc không đọc được dữ liệu.")

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = get_tempo(y, sr)   # ĐÃ SỬA: dùng beat_track, khớp lúc train

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    feats = {
        "zcr_mean": np.mean(zcr),
        "rms_mean": np.mean(rms),
        "tempo": tempo,
        "onset_mean": np.mean(onset_env),
        "onset_std": np.std(onset_env),
        "spec_centroid_mean": np.mean(spec_centroid),
        "spec_bandwidth_mean": np.mean(spec_bandwidth),
        "spec_rolloff_mean": np.mean(spec_rolloff),
    }
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        feats[f"mfcc{i+1}_std"] = np.std(mfcc[i])
    feats["chroma_mean"] = np.mean(chroma)

    return np.array([feats[name] for name in FEATURE_ORDER], dtype=float).reshape(1, -1)


# ============================================================
# 4. DỰ ĐOÁN THỰC TẾ
# ĐÃ SỬA: threshold về 0.5 (ngưỡng chuẩn) để không che giấu vấn đề gốc.
# Thêm debug in ra prob_ai thô để chẩn đoán model có bị lệch hay không.
# ============================================================
def predict_label(file_path: str, model, scaler, threshold: float = 0.5):
    features = extract_features(file_path)           # (1, 35)
    features_scaled = scaler.transform(features)      # BẮT BUỘC scale trước

    proba = model.predict_proba(features_scaled)[0]
    prob_ai = proba[1]  # Xác suất mô hình đoán là Nhạc AI (Nhãn 1)

    if DEBUG_MODE:
        print(f"[DEBUG] {os.path.basename(file_path)}: prob_ai = {prob_ai:.4f}")

    if prob_ai >= threshold:
        pred_class = 1
        confidence = prob_ai * 100
    else:
        pred_class = 0
        confidence = (1 - prob_ai) * 100

    return pred_class, confidence, prob_ai


# ============================================================
# 5. GIAO DIỆN NGƯỜI DÙNG
# ============================================================
st.title("🎵 Hệ thống Phân loại Nhạc AI vs Nhạc Thật")
st.markdown(
    "Tải lên file **ZIP** chứa các bản nhạc (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.aac`). "
    "Hệ thống sẽ giải nén, trích xuất đặc trưng âm thanh và dùng **model Machine Learning "
    "đã train** để dự đoán từng bài là **🤖 Nhạc AI** hay **🎤 Nhạc do nhạc sĩ thật tạo**, "
    "kèm theo độ tự tin (%)."
)

# Cho phép chỉnh ngưỡng ngay trên giao diện để dễ thử nghiệm, không cần sửa code
with st.sidebar:
    st.header("⚙️ Cài đặt")
    threshold_ui = st.slider(
        "Ngưỡng xác suất để kết luận là AI",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    show_debug = st.checkbox("Hiện xác suất thô (prob_ai) trên giao diện", value=True)

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"❌ Không thể tải model/scaler: {e}")
    st.stop()

uploaded_zip = st.file_uploader("📁 Chọn file ZIP chứa audio", type=["zip"])

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)
        except zipfile.BadZipFile:
            st.error("❌ File ZIP không hợp lệ hoặc đã bị hỏng. Vui lòng thử lại với file khác.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi khi giải nén file ZIP: {e}")
            st.stop()

        audio_paths = []
        for root, _, files in os.walk(tmp_dir):
            for fname in files:
                if fname.lower().endswith(AUDIO_EXTENSIONS):
                    audio_paths.append(os.path.join(root, fname))

        if not audio_paths:
            st.warning("⚠️ Không tìm thấy file audio nào trong file ZIP này.")
        else:
            audio_paths = sorted(audio_paths, key=lambda p: os.path.basename(p).lower())
            st.success(f"✅ Đã tìm thấy **{len(audio_paths)}** file audio. Đang phân tích...")
            st.markdown("---")
            st.subheader("📋 Kết quả phân loại")

            progress = st.progress(0)
            prob_list = []  # lưu lại toàn bộ prob_ai để xem thống kê tổng quan cuối trang

            for idx, path in enumerate(audio_paths):
                fname = os.path.basename(path)

                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.markdown(f"**🎧 {fname}**")
                    try:
                        with open(path, "rb") as audio_file:
                            st.audio(audio_file.read())
                    except Exception:
                        st.caption("(Không thể phát thử file này)")

                with col_right:
                    try:
                        pred_class, confidence, prob_ai = predict_label(
                            path, model, scaler, threshold=threshold_ui
                        )
                        prob_list.append(prob_ai)

                        label = LABEL_MAP[pred_class]
                        color = LABEL_COLOR[pred_class]
                        st.markdown(
                            f"<div style='padding:10px 0'>"
                            f"<span style='color:{color};font-weight:700;font-size:1.05em'>{label}</span><br>"
                            f"<span style='color:gray;font-size:0.9em'>Độ tự tin: {confidence:.1f}%</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.progress(min(max(confidence / 100, 0.0), 1.0))

                        if show_debug:
                            st.caption(f"🔍 prob_ai thô (trước ngưỡng): {prob_ai:.3f}")

                    except Exception as e:
                        st.markdown(
                            f"<span style='color:#f39c12'>⚠️ Không trích xuất được đặc trưng: {e}</span>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                progress.progress((idx + 1) / len(audio_paths))

            # Thống kê tổng quan prob_ai — giúp phát hiện nhanh nếu model bị lệch hệ thống
            if prob_list:
                st.subheader("📊 Thống kê xác suất AI (prob_ai) toàn bộ playlist")
                prob_arr = np.array(prob_list)
                col1, col2, col3 = st.columns(3)
                col1.metric("Trung bình prob_ai", f"{prob_arr.mean():.3f}")
                col2.metric("Nhỏ nhất", f"{prob_arr.min():.3f}")
                col3.metric("Lớn nhất", f"{prob_arr.max():.3f}")

                if prob_arr.mean() > 0.7:
                    st.warning(
                        "⚠️ Trung bình prob_ai toàn playlist khá cao (>0.7) — nếu playlist này "
                        "chủ yếu là nhạc thật, đây là dấu hiệu model đang bị lệch (thiên vị đoán AI), "
                        "cần xem lại dữ liệu huấn luyện hoặc cách trích đặc trưng."
                    )

st.caption(
    "Model: Random Forest — 35 đặc trưng âm thanh trích xuất bằng librosa "
    "(ZCR, RMS, tempo, onset, spectral centroid/bandwidth/rolloff, 13 MFCC mean+std, chroma)."
)
