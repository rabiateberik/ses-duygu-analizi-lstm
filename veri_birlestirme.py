import librosa
import numpy as np
import os
import glob

#RAVDESS, TESS ve SAVEE gibi farklı yapılardaki veri setlerini topladım, hepsini MFCC yöntemiyle sayısal verilere dönüştürdüm,
#etiketlerini standartlaştırdım ve LSTM modeline girecek şekilde hazır hale getirip kaydettim
# verileri LSTM için hazırlayan kodlar

# Klasör yolları (DuyguAnalizi klasörünün içindeyken çalışacak)
RAVDESS_PATH = "Verisetleri/RAVDESS"
TESS_PATH = "Verisetleri/TESS"
SAVEE_PATH = "Verisetleri/SAVEE"

# Duygu Sözlüğü, duygular veri setlerinde farklı şekilde etiketlendiği için burada normalizasyon yapıyoruz
duygu_sozlugu = {
    '01': 'neutral', 'n': 'neutral', 'neutral': 'neutral',
    '02': 'calm',    # Calm ve Neutral birbirine çok benzediği için birleştireceğiz
    '03': 'happy', 'h': 'happy', 'happy': 'happy',
    '04': 'sad', 'sa': 'sad', 'sad': 'sad',
    '05': 'angry', 'a': 'angry', 'angry': 'angry',
    '06': 'fear', 'f': 'fear', 'fear': 'fear',
    '07': 'disgust', 'd': 'disgust', 'disgust': 'disgust',
    '08': 'surprise', 'su': 'surprise', 'ps': 'surprise' 
}
# sesi sayıya çeviren fonksiyon
def ozellik_cikar(dosya_yolu):
    # Sesi yükle (Tüm verileri 22050 Hz'e sabitliyoruz)
    # duration=2.5 -> verileri eşitlemek için her ses dosyasının sadece 2.5 saniyesini alıyoruz 
    X, sample_rate = librosa.load(dosya_yolu, res_type='kaiser_fast', duration=2.5, offset=0.5)

    # MFCC Çıkar (40 özellik),sesin karakterini en iyi anlatan 40 temel özelliği çıkarıyoruz - LSTM'in kullanacağı ana veri bu
    # MFCC, insan kulağının sesi duyma şeklini taklit eden matematiksel bir yöntemdir
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    
    # Özelliklerin ortalamasını alarak boyutu sabitliyoruz
    return np.mean(mfccs.T, axis=0)

X = []
y = []

print(" Veri işleme başladı... Biraz sürebilir, bekleyelim.")

# ---------------------------------------------------------
# 1. RAVDESS YÜKLEME
# ---------------------------------------------------------
print(f"--- RAVDESS Yükleniyor... ---")
ravdess_dosyalar = glob.glob(os.path.join(RAVDESS_PATH, "**/*.wav"), recursive=True)

if not ravdess_dosyalar:
    print("UYARI: RAVDESS klasöründe dosya bulunamadı! Klasör yolunu kontrol et.")

for dosya in ravdess_dosyalar:
    dosya_adi = os.path.basename(dosya)
    # Dosya isimlendirmesi: 03-01-06-01... (6. rakam duyguyu verir)
    parcalar = dosya_adi.split("-")
    
    # Dosya ismi doğru formatta mı kontrol
    if len(parcalar) > 2:
        duygu_kodu = parcalar[2]
        
        if duygu_kodu in duygu_sozlugu:
            duygu = duygu_sozlugu[duygu_kodu]
            if duygu == 'calm': duygu = 'neutral' # Calm -> Neutral yap 
            
            ozellik = ozellik_cikar(dosya)
            X.append(ozellik)
            y.append(duygu)

# ---------------------------------------------------------
# 2. TESS YÜKLEME (Sadece YAF)
# ---------------------------------------------------------
print(f"--- TESS (YAF) Yükleniyor... ---")
tess_dosyalar = glob.glob(os.path.join(TESS_PATH, "**/*.wav"), recursive=True)

if not tess_dosyalar:
    print("UYARI: TESS klasöründe dosya bulunamadı!")
# Bu veri setinde hem yaşlı hem genç sesi vardı ben sadece genç(YAF) olanları alacak şekilde filtreledim
for dosya in tess_dosyalar:
    if "YAF_" in dosya: # Sadece genç sesleri alınıyor
        klasor_adi = os.path.basename(os.path.dirname(dosya))
        ham_duygu = klasor_adi.split("_")[1].lower()
        if ham_duygu == 'ps': ham_duygu = 'surprise'
        
        if ham_duygu in duygu_sozlugu:
            ozellik = ozellik_cikar(dosya)
            X.append(ozellik)
            y.append(ham_duygu)

# ---------------------------------------------------------
# 3. SAVEE YÜKLEME
# ---------------------------------------------------------
print(f"--- SAVEE Yükleniyor... ---")
savee_dosyalar = glob.glob(os.path.join(SAVEE_PATH, "*.wav"))

if not savee_dosyalar:
    print("UYARI: SAVEE klasöründe dosya bulunamadı!")

for dosya in savee_dosyalar:
    dosya_adi = os.path.basename(dosya)
    ham_duygu = ""
    # Harflere göre duygu tahmini
    if dosya_adi.startswith('a'): ham_duygu = 'angry'
    elif dosya_adi.startswith('d'): ham_duygu = 'disgust'
    elif dosya_adi.startswith('f'): ham_duygu = 'fear'
    elif dosya_adi.startswith('h'): ham_duygu = 'happy'
    elif dosya_adi.startswith('n'): ham_duygu = 'neutral'
    elif dosya_adi.startswith('sa'): ham_duygu = 'sad'
    elif dosya_adi.startswith('su'): ham_duygu = 'surprise'
    
    if ham_duygu != "":
        ozellik = ozellik_cikar(dosya)
        X.append(ozellik)
        y.append(ham_duygu)

# ---------------------------------------------------------
# KAYIT
# ---------------------------------------------------------
#listeleri array e çeviriyoruz
X = np.array(X)
y = np.array(y)

print("\n✅ İŞLEM TAMAMLANDI!")
print(f"Toplam Veri Sayısı: {len(X)}")
print(f"Örnek Veri Şekli: {X.shape}")

np.save("X_ozellikler.npy", X)# içinde sesleirn sayısal halleri tutuluyor
np.save("y_etiketler.npy", y)#içinde o seslerin hangi duygu olduğu yazıyor
print(" Dosyalar kaydedildi: X_ozellikler.npy ve y_etiketler.npy")