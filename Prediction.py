import pickle
import streamlit as st

model = pickle.load(open("stroke.sav", "rb"))

st.title("Prediksi Pasien Stroke")

st.write("Masukkan data pasien dibawah ini")

gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
if gender == "Laki-laki":
    gender = 0
else:
    gender = 1
age = st.selectbox(
    "Umur",
    [
        "Kategori 1 (<=20 tahun)",
        "Kategori 2 (21-40 tahun)",
        "Kategori 3 (41-60 tahun)",
        "Kategori 4 (>60 tahun)",
    ],
)
if age == "Kategori 1 (<=20 tahun)":
    age = 0
elif age == "Kategori 2 (21-40 tahun)":
    age = 1
elif age == "Kategori 3 (41-60 tahun)":
    age = 2
else:
    age = 3
hypertension = st.selectbox("Memiliki Hipertensi", ["Tidak", "Ya"])
if hypertension == "Tidak":
    hypertension = 0
else:
    hypertension = 1
heart_disease = st.selectbox("Memiliki Riwayat Penyakit Jantung", ["Tidak", "Ya"])
if heart_disease == "Tidak":
    heart_disease = 0
else:
    heart_disease = 1
ever_married = st.selectbox("Pernah Menikah", ["Sudah", "Belum"])
if ever_married == "Sudah":
    ever_married = 0
else:
    ever_married = 1
work_type = st.selectbox(
    "Tipe Pekerjaan",
    ["Belum pernah bekerja", "Anak-anak", "PNS", "Pekerja Swasta", "Wiraswasta"],
)
if work_type == "Belum pernah bekerja":
    work_type = 0
elif work_type == "Anak-anak":
    work_type = 1
elif work_type == "PNS":
    work_type = 2
elif work_type == "Pekerja Swasta":
    work_type = 3
else:
    work_type = 4
Residence_type = st.selectbox("Tipe Tempat Tinggal", ["Perkotaan", "Pedesaan"])
if Residence_type == "Perkotaan":
    Residence_type = 0
else:
    Residence_type = 1
avg_glucose_level = st.selectbox(
    "Rata-rata Kadar Glukosa",
    [
        "Kategori 1 (kadar <=77.07)",
        "Kategori 2 (kadar 77.08-91.68)",
        "Kategori 3 (kadar 91.69-113.57)",
        "Kategori 4 (kadar >113.57)",
    ],
)
if avg_glucose_level == "Kategori 1 (kadar <=77.07)":
    avg_glucose_level = 0
elif avg_glucose_level == "Kategori 2 (kadar 77.08-91.68)":
    avg_glucose_level = 1
elif avg_glucose_level == "Kategori 3 (kadar 91.69-113.57)":
    avg_glucose_level = 3
else:
    avg_glucose_level = 4
bmi = st.selectbox(
    "BMI (Body Mass Index)",
    [
        "Kategori 1 (<=23.5)",
        "Kategori 2 (23.6-28.1)",
        "Kategori 3 (28.2-33.1)",
        "Kategori 4 (>33.1)",
    ],
)
if bmi == "Kategori 1 (<=23.5)":
    bmi = 0
elif bmi == "Kategori 2 (23.6-28.1)":
    bmi = 1
elif bmi == "Kategori 3 (28.2-33.1)":
    bmi = 2
else:
    bmi = 3
smoking_status = st.selectbox(
    "Status Merokok",
    ["Tidak Diketahui", "Tidak Pernah", "Pernah Merokok", "Aktif Merokok"],
)
if smoking_status == "Tidak Diketahui":
    smoking_status = 0
elif smoking_status == "Tidak Pernah":
    smoking_status = 1
elif smoking_status == "Pernah Merokok":
    smoking_status = 2
else:
    smoking_status = 3


if st.button("Prediksi"):
    X = [
        [
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            Residence_type,
            avg_glucose_level,
            bmi,
            smoking_status,
        ]
    ]
    hasil = model.predict(X)
    if hasil[0] == 0:
        st.write("Pasien tidak memiliki kemungkinan untuk mengidap Stroke")
        print(hasil[0])
    else:
        st.write("Pasien memiliki kemungkinan besar untuk mengidap Stroke")
        print(hasil[0])
