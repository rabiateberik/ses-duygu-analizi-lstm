import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🥧 Veri seti pasta grafiği hazırlanıyor...")

# --- DOSYALARI YÜKLE  ---
try:
    # etiketleri yükle
    y_tum_veri = np.load("y_etiketler.npy")
    
    # 2. Sınıf isimlerini yükle
    if os.path.exists("siniflar.npy"):
        siniflar = np.load("siniflar.npy")
    else:
        print("⚠️ 'siniflar.npy' bulunamadı, varsayılan isimler kullanılıyor.")
        siniflar = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])

    toplam_sayi = len(y_tum_veri)
    print(f"✅ Toplam Veri Sayısı: {toplam_sayi}")

except Exception as e:
    print(f"❌ Hata: Dosyalar okunamadı. ({e})")
    exit()

# --- VERİYİ DÜZENLE ---
# One-Hot ise normale çevir
if y_tum_veri.ndim > 1:
    y_indices = np.argmax(y_tum_veri, axis=1)
else:
    y_indices = y_tum_veri

# İsimlere çevir
try:
    y_labels = siniflar[y_indices]
except IndexError:
    y_labels = y_indices

# Sayım yap
unique, counts = np.unique(y_labels, return_counts=True)
duygu_sozlugu = dict(zip(unique, counts))

# --- PASTA GRAFİĞİ ÇİZ ---
plt.figure(figsize=(10, 10)) 

# Renk paleti seçelim 
colors = sns.color_palette('pastel')[0:len(duygu_sozlugu)]

explode = [0.03] * len(duygu_sozlugu) 

# Grafiği çizdir
# autopct='%1.1f%%': Yüzdeleri virgülden sonra 1 basamak gösterir (örn: 14.5%)
plt.pie(counts, labels=unique, colors=colors, autopct='%1.1f%%', 
        startangle=140, pctdistance=0.85, explode=explode,
        textprops={'fontsize': 14, 'fontweight': 'bold'})


centre_circle = plt.Circle((0,0),0.60,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title(f"Tüm Veri Seti Duygu Dağılımı\n(Toplam {toplam_sayi} Ses)", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig("veri_seti_pasta_grafigi.png")
print("✅ Pasta grafiği 'veri_seti_pasta_grafigi.png' olarak kaydedildi!")
plt.show()