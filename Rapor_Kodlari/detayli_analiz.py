import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from itertools import cycle

# --- AYARLAR ---
sns.set_style("whitegrid") # Arka planı ızgaralı yap, şık dursun

# 1. VERİLERİ YÜKLE VE AYNI ŞEKİLDE BÖL
print("📦 Veriler hazırlanıyor...")
try:
    X = np.load("X_ozellikler.npy")
    y = np.load("y_etiketler.npy")
except:
    print("❌ Dosyalar bulunamadı! İsimleri kontrol et.")
    exit()

# Etiketleri hazırla
lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
siniflar = lb.classes_
n_classes = len(siniflar)

# Test verisini ayır (random_state=42 ile eğitimdekiyle AYNI veriyi alıyoruz)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_test = np.expand_dims(X_test, axis=2) # LSTM boyutu

# 2. MODELİ YÜKLE VE TAHMİN AL
print("🧠 Model tahmin yapıyor...")
model = load_model("duygu_modeli_yeni.h5")
y_pred_probs = model.predict(X_test, verbose=0) # Olasılıklar 
y_pred_classes = np.argmax(y_pred_probs, axis=1) # Sınıf tahmini 
y_true_classes = np.argmax(y_test, axis=1) # Gerçek sınıflar

# --- GRAFİK 1: ROC EĞRİSİ  ---
print("📈 ROC Eğrisi çiziliyor...")

# Her sınıf için ROC hesapla
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Çizim
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='{0} (AUC = {1:0.2f})'.format(siniflar[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2) # Rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm)')
plt.ylabel('True Positive Rate (Doğru Tespit)')
plt.title('Her Duygu İçin ROC Eğrisi ve AUC Başarısı', fontsize=15)
plt.legend(loc="lower right")
plt.show()

# --- GRAFİK 2: EN ÇOK KARIŞTIRILAN DUYGULAR (HATA ANALİZİ) ---
print("📊 Hata analizi yapılıyor...")

# Sadece hatalı tahminleri bulalım
hatalar = []
for gercek, tahmin in zip(y_true_classes, y_pred_classes):
    if gercek != tahmin:
        pair_name = f"{siniflar[gercek]} -> {siniflar[tahmin]}"
        hatalar.append(pair_name)

# Hataları say
if len(hatalar) > 0:
    hata_serisi = pd.Series(hatalar).value_counts().head(10) # En çok yapılan 10 hata

    plt.figure(figsize=(12, 6))
    sns.barplot(x=hata_serisi.values, y=hata_serisi.index, palette="rocket")
    
    plt.title('Modelin En Çok Yaptığı 10 Hata (Gerçek -> Tahmin)', fontsize=15)
    plt.xlabel('Hata Sayısı')
    
    # Çubukların ucuna sayıları yaz
    for i, v in enumerate(hata_serisi.values):
        plt.text(v + 0.1, i, str(v), color='black', fontweight='bold', va='center')
        
    plt.tight_layout()
    plt.show()
else:
    print("🎉 İnanılmaz! Hiç hata yok, bu grafik çizilemiyor.")

print("✅ Tüm grafikler hazırlandı.")