import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Sequential # <-- Sequential'ı buraya ekledik

# --- AYARLAR ---
sns.set_style("darkgrid")

# 1. VERİLERİ YÜKLE
print("📦 Veriler hazırlanıyor...")
try:
    # Dosya isimlerin neyse onları yaz (X_ozellikler veya X.npy)
    X = np.load("X_ozellikler.npy")
    y = np.load("y_etiketler.npy")
except:
    print("❌ Dosyalar bulunamadı!")
    exit()

# Etiketleme ve Bölme
lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
siniflar = lb.classes_

# random_state=42 sayesinde eğitimdekiyle AYNI veriyi alıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_test_lstm = np.expand_dims(X_test, axis=2) # Model boyutu

# 2. MODELİ YÜKLE
print("🧠 Model yükleniyor...")
try:
    model = load_model("duygu_modeli_yeni.h5")
except:
    print("❌ Model dosyası bulunamadı!")
    exit()

# --- GRAFİK 1: t-SNE KÜMELEME ---
print("🌌 t-SNE hesaplanıyor... (Model ısınıyor...)")

# --- [DÜZELTME BURADA] ---
# Eski yöntem hata verdiği için, son katmanı atıp yeni bir model yapıyoruz.
# Bu yöntem 'model.input' kullanmadığı için hata vermez.
feature_extractor = Sequential(model.layers[:-1]) 
# -------------------------

# Artık özellikleri çıkarabiliriz
features = feature_extractor.predict(X_test_lstm, verbose=0)

# 2 Boyuta indirgeme (t-SNE)
print("   -> Boyut indirgeniyor (Bu işlem 10-20 sn sürer)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(features)

# Gerçek sınıfları isme çevir
y_test_labels = lb.inverse_transform(np.argmax(y_test, axis=1))

# Çizim
print("🎨 Grafik çiziliyor...")
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y_test_labels, palette="bright", s=60, alpha=0.8)
plt.title('t-SNE: Model Duyguları Nasıl Grupluyor?', fontsize=16)
plt.legend(title='Duygular', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- GRAFİK 2: KENDİNE GÜVEN (CONFIDENCE) ---
print("💪 Güven analizi yapılıyor...")

y_pred_probs = model.predict(X_test_lstm, verbose=0)
max_probs = np.max(y_pred_probs, axis=1)
pred_classes = np.argmax(y_pred_probs, axis=1)
pred_labels = lb.inverse_transform(pred_classes)

df_conf = pd.DataFrame({'Duygu': pred_labels, 'Guven': max_probs})

plt.figure(figsize=(12, 6))
sns.boxplot(x='Duygu', y='Guven', data=df_conf, palette="viridis")
sns.stripplot(x='Duygu', y='Guven', data=df_conf, color='black', alpha=0.3, size=2)

plt.title('Modelin Tahmin Güveni', fontsize=16)
plt.ylabel('Güven Skoru (0.0 - 1.0)', fontsize=12)
plt.xlabel('Tahmin Edilen Duygu', fontsize=12)
plt.ylim(0.5, 1.02)
plt.show()

print("✅ Şov bitti. Geçmiş olsun!")