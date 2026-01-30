import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# 1. VERİLERİ YÜKLE 
print("📦 Veriler yükleniyor...")
try:
    # Dosya isimlerin X.npy ve y.npy ise burayı düzelt!
    X = np.load("X_ozellikler.npy") 
    y = np.load("y_etiketler.npy")
except FileNotFoundError:
    print("❌ HATA: Veri dosyaları (X_ozellikler.npy vb.) bulunamadı!")
    exit()

# 2. AYNI ŞEKİLDE BÖL (random_state=42 sayesinde tıpatıp aynısı olur)
print("✂️ Test verisi ayrıştırılıyor...")
lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
siniflar = lb.classes_

# İşte burası modelin eğitilirken kullandığı test setinin aynısını verir
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# LSTM formatına uygun hale getir (Boyut ekleme)
X_test = np.expand_dims(X_test, axis=2)

# 3. KAYITLI MODELİ YÜKLE
print("🧠 Eğitilmiş model yükleniyor...")
try:
    model = load_model("duygu_modeli_yeni.h5") 
except:
    print("❌ HATA: 'duygu_modeli_yeni.h5' dosyası bulunamadı! Modelin kaydedilmemiş olabilir.")
    exit()

# 4. TAHMİN YAP
print("🎯 Tahminler yapılıyor...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 5. MATRİSİ ÇİZ
print("🎨 Grafik çiziliyor...")
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
# Kutucuklara sadece sayı yazalım
labels = ["{0:0.0f}".format(value) for value in cm.flatten()]
labels = np.asarray(labels).reshape(cm.shape[0], cm.shape[1])

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
            xticklabels=siniflar, yticklabels=siniflar, cbar=False)

accuracy = np.trace(cm) / np.sum(cm)
plt.title(f'GERÇEK Test Sonuçları (Accuracy: %{accuracy*100:.2f})', fontsize=14)
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.show()

# 6. raporu yazdır
print("\n" + "="*40)
print(f"🏆 TEST BAŞARISI: %{accuracy*100:.2f}")
print("="*40)
print(classification_report(y_true_classes, y_pred_classes, target_names=siniflar))