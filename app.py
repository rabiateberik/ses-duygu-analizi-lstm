import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Ses Duygu Analizi", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modelin İngilizce çıktılarını burada Türkçeye çeviriyoruz.
CEVIRI = {
    'neutral': 'Nötr 😐',
    'calm': 'Sakin 😌',
    'happy': 'Mutlu 😃',
    'sad': 'Üzgün 😔',
    'angry': 'Kızgın 😡',
    'fear': 'Korku 😱',
    'disgust': 'İğrenme 🤢',
    'surprise': 'Şaşkın 😲'
}

# CSS kısmı
st.markdown("""
<style>
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    /* Başlık */
    h1 {
        color: #00d4ff;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.7);
    }
    
    /* Alt Başlık */
    .subtitle {
        text-align: center;
        color: #bdc3c7;
        font-size: 20px;
        margin-bottom: 40px;
    }

    /* Buton */
    .stButton>button {
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(28, 181, 224, 0.6);
    }

    /* Sonuç Kartı */
    .result-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def model_yukle():
    # Model dosyasını yükle
    if os.path.exists('duygu_modeli_yeni.h5'):
        return tf.keras.models.load_model('duygu_modeli_yeni.h5')
    else:
        return None

@st.cache_data
def siniflari_yukle():
    try:
        return np.load('siniflar.npy')
    except:
        # Eğer dosya yoksa varsayılan İngilizce listeyi döndür 
        return np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

def tahmin_et(dosya_yolu, model, classes):
    # 1. Sesi Yükle
    audio, sr = librosa.load(dosya_yolu, res_type='kaiser_fast', duration=2.5, offset=0.5)
    
    # 2. MFCC Çıkar
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # 3. Boyut Artır
    input_data = np.expand_dims(mfccs_mean, axis=0) # (1, 40)
    input_data = np.expand_dims(input_data, axis=2) # (1, 40, 1)
    
    # 4. Tahmin
    tahminler = model.predict(input_data)
    idx = np.argmax(tahminler)
    
    return classes[idx], np.max(tahminler) * 100, tahminler[0]

# --- ARAYÜZ BAŞLANGICI ---

# Başlıklar
st.markdown("<h1>🧠 Ses Duygu Analizi Sistemi</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>LSTM ile Ses Analizi</div>", unsafe_allow_html=True)

# Yan Menü
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637107.png", width=100)
    st.title("Proje Hakkında")
    st.info("Bu proje **LSTM** mimarisi kullanılarak geliştirilmiştir.")
    st.write("📊 **Model Başarısı:** %92.14")
    st.write("📂 **Veri Seti:** TESS + RAVDESS + SAVEE")
    st.divider()
    st.caption("Geliştiren: [Senin Adın]")

# Modeli Yükle
model = model_yukle()
classes = siniflari_yukle()

if model is None:
    st.error("❌ HATA: 'duygu_modeli_yeni.h5' dosyası bulunamadı! Lütfen dosyanın proje klasöründe olduğundan emin ol.")
    st.stop()

# --- ANA ALAN ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Ses Dosyası Seç")
    st.write("Analiz etmek istediğin `.wav` dosyasını buraya sürükle.")
    
    uploaded_file = st.file_uploader("", type=["wav"])
    
    if uploaded_file:
        # Dosyayı geçici kaydet
        with open("temp_upload.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Dosya başarıyla yüklendi! ✅")
        st.audio("temp_upload.wav")
        
        if st.button("ANALİZ ET 🚀", key="analyze_btn"):
             with st.spinner('Yapay zeka sesi dinliyor...'):
                try:
                    sonuc, guven, dagilim = tahmin_et("temp_upload.wav", model, classes)
                    
                    # Sonuçları Session State'e kaydet
                    st.session_state['sonuc'] = sonuc
                    st.session_state['guven'] = guven
                    st.session_state['dagilim'] = dagilim
                    st.session_state['analiz_yapildi'] = True
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")

with col2:
    if 'analiz_yapildi' in st.session_state and st.session_state['analiz_yapildi']:
        sonuc = st.session_state['sonuc']
        guven = st.session_state['guven']
        dagilim = st.session_state['dagilim']
        
        # --- TÜRKÇELEŞTİRME KISMI ---
        # 1. Tahmin edilen tek kelimeyi çevir (Örn: 'happy' -> 'Mutlu 😃')
        # .lower() kullanarak küçük harfe çeviriyoruz ki eşleşme garanti olsun.
        sonuc_tr = CEVIRI.get(sonuc.lower(), sonuc).upper() 
        
        # 2. Grafik etiketlerini topluca çevir
        classes_tr = [CEVIRI.get(c.lower(), c) for c in classes]
        
        # Sonuç Kartı
        st.markdown(f"""
        <div class='result-box'>
            <h3 style='margin:0; color:#bdc3c7;'>Tespit Edilen Duygu</h3>
            <h1 style='font-size: 50px; margin: 10px 0; color: #f1c40f; text-shadow: 0 0 10px #f1c40f;'>{sonuc_tr}</h1>
            <p style='font-size: 18px;'>Güven Oranı: <span style='color:#2ecc71; font-weight:bold;'>%{guven:.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Boşluk
        st.markdown("### 📊 Olasılık Dağılımı")
        
        # Türkçe etiketlerle grafik çiz
        chart_data = dict(zip(classes_tr, dagilim))
        st.bar_chart(chart_data)

    else:
        # Boşken görünecek yazı
        st.info("👈 Analiz sonucunu görmek için soldan bir dosya yükleyip butona bas.")