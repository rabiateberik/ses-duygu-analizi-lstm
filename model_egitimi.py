import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. veriler yükleniyor ve 
print("📦 Veriler yükleniyor...")
try:
    X = np.load("X_ozellikler.npy")
    y = np.load("y_etiketler.npy")
except FileNotFoundError:
    print("❌ HATA: .npy dosyaları (X_ozellikler, y_etiketler) bulunamadı!")
    exit()

# 2. etiketler sayıya çevriliyor
lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
siniflar = lb.classes_

# 3.veriyi %80 eğitim %20 test olacak şekilde ayırdık
print("Veriler ayrıştırılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# test verileri kaydediliyor
print("💾 Test verileri 'X_test_gercek.npy' olarak kaydediliyor...")
np.save("X_test_gercek.npy", X_test)
np.save("y_test_gercek.npy", y_test)
np.save("siniflar.npy", siniflar) 

# 4.boyut ayarlanıyor
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 5. MODEL MİMARİSİ
model = Sequential()
# 1. Katman: Sesteki zaman akışını yakalanıyor
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))#ezber bozmak için %30 unutma 
# 2. Katman: Bilgiyi özetler
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
# 3. Katman: Karar Verme (Dense)
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
# 4. Çıkış Katmanı: Sonuç  - 7 Duygu sınıfı için olasılık verir.
model.add(Dense(len(siniflar), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# en yüksek başarıyı yakaladığında kaydediliyor
checkpoint = ModelCheckpoint("duygu_modeli_yeni.h5", 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

# 6. eğitim başlatılıyor
print("Eğitim başlıyor...")
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint], 
                    verbose=1)

# grafik çiz
print(" Başarı grafikleri çiziliyor...")

plt.figure(figsize=(12, 5))

# Doğruluk Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim (Train)')
plt.plot(history.history['val_accuracy'], label='Test (Validation)')
plt.title('Model Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Kayıp Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Test Kaybı')
plt.title('Loss (Kayıp) Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig("basarim_grafigi.png")
print("İŞLEM TAMAM: Model eğitildi, test verileri kaydedildi.")