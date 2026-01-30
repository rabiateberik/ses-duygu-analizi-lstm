import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import pandas as pd
import os

# --- AYARLAR VE YÜKLEME ---
print("📦 Test Verileri ve Model yükleniyor...")

try:
    # 1. SADECE TEST VERİLERİNİ YÜKLE 
    X_test = np.load('X_test_gercek.npy')
    y_test = np.load('y_test_gercek.npy')
    
    # Sınıf isimlerini yükle
    if os.path.exists("siniflar.npy"):
        sinif_isimleri = np.load("siniflar.npy")
    else:
        sinif_isimleri = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # 2. Modeli Yükle
    model = load_model('duygu_modeli_yeni.h5') # Model ismin doğru mu kontrol et
    print("✅ Veriler hazır.")

except Exception as e:
    print(f"❌ Hata: Dosyalar eksik. ({e})")
    exit()

# --- TAHMİN ---
print("🧠 Model sınava giriyor (Test)...")

# LSTM Tahmini
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1) # One-hot'tan sayıya çevir

# --- GRAFİK 1: CONFUSION MATRIX (SADE) ---
print("🎨 Grafik 1: Karmaşıklık Matrisi çiziliyor...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
# Sadece sayıları yazdırıyoruz (Yüzde yok)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sinif_isimleri, yticklabels=sinif_isimleri, cbar=False)
plt.title('Confusion Matrix (Test Verisi)', fontsize=16, fontweight='bold')
plt.ylabel('Gerçek Sınıflar')
plt.xlabel('Tahmin Edilen')
plt.tight_layout()
plt.show()

# --- GRAFİK 2: DETAYLI KARNE (HEATMAP) ---
print("📊 Grafik 2: Performans Karnesi hazırlanıyor...")

report_dict = classification_report(y_true, y_pred, target_names=sinif_isimleri, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
# Ortalama satırlarını at, sadece duygular kalsın
df_report_classes = df_report.iloc[:-3, :3] 

plt.figure(figsize=(10, 6))
sns.heatmap(df_report_classes, annot=True, cmap='viridis', fmt='.2f', linewidths=1)
plt.title('Detaylı Performans Karnesi (Precision - Recall - F1)', fontsize=14, fontweight='bold')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# GRAFİK 3: BAŞARI SIRALAMASI 
print("🏆 Grafik 3: Başarı Sıralaması çiziliyor...")

class_accuracy = df_report_classes['f1-score'].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=class_accuracy.index, y=class_accuracy.values, palette='magma')

plt.title('Model Hangi Duyguda Daha Başarılı? (F1 Score)', fontsize=14, fontweight='bold')
plt.ylabel('Başarı Puanı (0-1 Arası)')
plt.xlabel('Duygular')
plt.ylim(0.0, 1.1) 

# Puanları sütunların üstüne yaz
for i, v in enumerate(class_accuracy.values):
    plt.text(i, v + 0.02, f"%{v*100:.1f}", ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()