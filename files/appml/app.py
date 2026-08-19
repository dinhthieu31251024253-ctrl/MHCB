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

DURATION = 20        # số giây audio đọc để trích xuất đặc trưng
SAMPLE_RATE = 22050   # tần số lấy mẫu
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")

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
    0: "#27ae60",  # xanh lá
    1: "#e74c3c",  # đỏ
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
# HÀM PHỤ: LẤY TEMPO ỔN ĐỊNH QUA CÁC PHIÊN BẢN LIBROSA KHÁC NHAU
# ============================================================
def get_tempo(y, sr, onset_env):
    """librosa đổi vị trí hàm tempo giữa các phiên bản (beat.tempo ->
    feature.rhythm.tempo), nên thử lần lượt nhiều cách để đảm bảo tương thích."""
    # Cách 1: API mới (librosa >= 0.10)
    try:
        return float(librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        pass
    # Cách 2: API cũ (librosa < 0.10)
    try:
        return float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        pass
    # Cách 3: fallback qua beat_track
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
        return float(tempo)
    except Exception:
        return 0.0


# ============================================================
# 2 & 3. TRÍCH XUẤT 35 ĐẶC TRƯNG ÂM THANH
# ============================================================
def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    if y is None or len(y) == 0:
        raise ValueError("File audio rỗng hoặc không đọc được dữ liệu.")

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = get_tempo(y, sr, onset_env)

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
# ============================================================
def predict_label(file_path: str, model, scaler):
    features = extract_features(file_path)          # (1, 35)
    features_scaled = scaler.transform(features)     # BẮT BUỘC scale trước
    pred_class = int(model.predict(features_scaled)[0])
    proba = model.predict_proba(features_scaled)[0]
    confidence = float(proba[pred_class]) * 100
    return pred_class, confidence


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

        # ---- Giải nén ----
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)
        except zipfile.BadZipFile:
            st.error("❌ File ZIP không hợp lệ hoặc đã bị hỏng. Vui lòng thử lại với file khác.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi khi giải nén file ZIP: {e}")
            st.stop()

        # ---- Quét toàn bộ file audio (kể cả thư mục con) ----
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
                        pred_class, confidence = predict_label(path, model, scaler)
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
                    except Exception as e:
                        st.markdown(
                            f"<span style='color:#f39c12'>⚠️ Không trích xuất được đặc trưng: {e}</span>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                progress.progress((idx + 1) / len(audio_paths))

st.caption(
    "Model: Random Forest / Logistic Regression — 35 đặc trưng âm thanh trích xuất bằng librosa "
    "(ZCR, RMS, tempo, onset, spectral centroid/bandwidth/rolloff, 13 MFCC mean+std, chroma). "
    "Kết quả mang tính tham khảo, phụ thuộc vào chất lượng và độ đa dạng của dữ liệu huấn luyện."
)
