import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization # <--- SimpleRNN eklendi
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. veriler yükleniyor
print("📦 Veriler yükleniyor...")
try:
    X = np.load("X_ozellikler.npy")
    y = np.load("y_etiketler.npy")
except FileNotFoundError:
    print("❌ HATA: .npy dosyaları bulunamadı!")
    exit()

# 2. etiketleme kısmı
lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
siniflar = lb.classes_

# 3. eğitim ve test bölünmesi
print(" Veriler ayrıştırılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# test verielri kontrol ediliyor
print(" Test verileri kontrol ediliyor...")
np.save("X_test_gercek.npy", X_test)
np.save("y_test_gercek.npy", y_test)
np.save("siniflar.npy", siniflar)

# 4. boyutlar ayarlanıyor
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 5. rnn model mimarisi
print("RNN Modeli inşa ediliyor...")
model = Sequential()

model.add(SimpleRNN(128, return_sequences=True, input_shape=(X_train.shape[1], 1))) 
model.add(Dropout(0.3))

model.add(SimpleRNN(64, return_sequences=False)) # İkinci katman
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len(siniflar), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint 
checkpoint = ModelCheckpoint("duygu_modeli_rnn.h5", 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

# 6. eğitim başlatılıyor
print(" RNN Eğitimi başlıyor...")
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint], 
                    verbose=1)

# --- GRAFİK ÇİZME ---
print("📊RNN Grafikleri çiziliyor...")

plt.figure(figsize=(12, 5))

# Doğruluk Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='RNN Eğitim (Train)', color='blue')
plt.plot(history.history['val_accuracy'], label='RNN Test (Validation)', color='orange')
plt.title('RNN Model Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Kayıp Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='RNN Eğitim Kaybı', color='red')
plt.plot(history.history['val_loss'], label='RNN Test Kaybı', color='green')
plt.title('RNN Loss (Kayıp) Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig("rnn_basarim_grafigi.png")
print("✅ İŞLEM TAMAM: RNN modeli eğitildi ve 'duygu_modeli_rnn.h5' olarak kaydedildi.")
plt.show()