# Stingles-Bee-Clasification
Penelitian ini bertujuan untuk membangun model klasifikasi citra lebah madu tanpa sengat menggunakan Convolutional Neural Network (CNN) berbasis InceptionResNetV2.  Model dilatih menggunakan teknik transfer learning untuk meningkatkan performa klasifikasi pada dataset terbatas.

## 📁 Struktur Project

app/              -> Flask API dan Streamlit App  
pre-processing/   -> Script preprocessing dataset  
src/              -> Training dan evaluasi model  

---

## 🧠 Arsitektur Model

Model menggunakan InceptionResNetV2 sebagai backbone CNN
dengan teknik transfer learning.

Input size: 299x299x3  
Output: 4 kelas spesies lebah  
Metode OOD: Energy-based score  

## 📊 Dataset

Dataset terdiri dari 4 spesies lebah tanpa sengat:

- Heterotrigona itama
- Tetrigona apicalis
- Tetrigona binghamii
- Tetrigona vidua

Dataset digunakan untuk training dan validation 
dengan teknik transfer learning.
