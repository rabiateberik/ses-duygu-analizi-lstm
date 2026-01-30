# 🎧 Ses Duygu Analizi (RNN & LSTM)

Bu proje, insan sesinden duygu analizi (Speech Emotion Recognition) yapmak amacıyla
geliştirilmiştir.  
Ses verileri üzerinden çıkarılan özellikler kullanılarak **RNN** ve **LSTM**
tabanlı derin öğrenme modelleri eğitilmiş ve performansları karşılaştırılmıştır.

## 🎯 Projenin Amacı

- İnsan sesinden duyguları otomatik olarak tanıyabilen modeller geliştirmek  
- Zaman serisi verileri üzerinde **RNN ve LSTM** yapılarını karşılaştırmak  
- Derin öğrenme yöntemlerinin ses verileri üzerindeki başarımını incelemek  

## 🧠 Kullanılan Yöntemler

- Özellik çıkarımı (MFCC vb.)
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- Çok sınıflı sınıflandırma
- Confusion Matrix ve doğruluk (accuracy) analizi

## 📊 Kullanılan Veri Setleri

Bu projede aşağıdaki açık kaynak ses duygu analizi veri setleri kullanılmıştır:

- **RAVDESS**
- **SAVEE**
- **TESS**

> Veri setleri dosya boyutlarının büyük olması nedeniyle GitHub reposuna eklenmemiştir.

## 🛠️ Kullanılan Teknolojiler

- Python  
- NumPy  
- Librosa  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  

## ⚙️ Proje Aşamaları

1. Ses verilerinin yüklenmesi  
2. Özellik çıkarımı  
3. Veri setinin eğitim ve test olarak ayrılması  
4. **RNN modelinin eğitilmesi**  
5. **LSTM modelinin eğitilmesi**  
6. Modellerin test edilmesi  
7. Sonuçların karşılaştırılması  

## 📈 Değerlendirme ve Karşılaştırma

Model performansları aşağıdaki metrikler kullanılarak değerlendirilmiştir:

- Doğruluk (Accuracy)
- Confusion Matrix

Elde edilen sonuçlara göre **LSTM modeli**, uzun dönem bağımlılıkları daha iyi
öğrendiği için RNN modeline kıyasla daha başarılı sonuçlar üretmiştir.

## 🚀 Proje Durumu

Proje çalışır durumdadır ve geliştirilmeye açıktır.  
İlerleyen aşamalarda CNN veya Transformer tabanlı modeller ile
karşılaştırmalar yapılabilir.


👤 **Geliştirici**  
- GitHub: (kullanıcı adını buraya ekleyebilirsin)
