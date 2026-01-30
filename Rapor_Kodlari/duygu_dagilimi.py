import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("📦 Genel Veri Dağılımı analiz ediliyor...")

# --- DOSYALARI YÜKLE ---
try:
    # 1. Bütün etiketleri yükle
    y_tum_veri = np.load("y_etiketler.npy")
    
    # 2. Sınıf isimlerini yükle 
    if os.path.exists("siniflar.npy"):
        siniflar = np.load("siniflar.npy")
    else:
        # Eğer sınıf dosyası yoksa biz manuel tanımlayalım 
        print("⚠️ 'siniflar.npy' bulunamadı, varsayılan isimler kullanılıyor.")
        siniflar = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])

    print(f"✅ Toplam Veri Sayısı: {len(y_tum_veri)}")

except Exception as e:
    print(f"❌ Hata: Dosyalar okunamadı. ({e})")
    print("Lütfen 'y_etiketler.npy' dosyasının klasörde olduğundan emin ol.")
    exit()

# --- VERİYİ DÜZENLE ---
if y_tum_veri.ndim > 1:
    y_indices = np.argmax(y_tum_veri, axis=1)
else:
    y_indices = y_tum_veri 

try:
    y_labels = siniflar[y_indices]
except IndexError:
    print("⚠️ Etiket sayıları ile sınıf listesi uyuşmuyor! Sadece sayıları kullanacağız.")
    y_labels = y_indices

# Sayım yap 
unique, counts = np.unique(y_labels, return_counts=True)
duygu_sozlugu = dict(zip(unique, counts))

print("📊 Duygu Sayıları:")
for k, v in duygu_sozlugu.items():
    print(f" - {k}: {v}")

# --- GRAFİK ÇİZ  ---
plt.figure(figsize=(12, 6))

# Renkli sütun grafiği
ax = sns.barplot(x=list(duygu_sozlugu.keys()), y=list(duygu_sozlugu.values()), palette="rocket")

plt.title(f"Proje Veri Seti Dağılımı (Toplam {len(y_tum_veri)} Ses)", fontsize=16, fontweight='bold')
plt.xlabel("Duygu Sınıfları", fontsize=12)
plt.ylabel("Ses Dosyası Sayısı", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Sütunların tepesine sayıları yaz
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("veri_seti_grafigi.png")
print("✅ Grafik 'veri_seti_grafigi.png' olarak kaydedildi! Raporuna koyabilirsin.")
plt.show()