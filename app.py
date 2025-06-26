import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier # Pastikan import ini ada

# Muat model pipeline yang sudah dilatih
# Pastikan file .pkl ada di direktori yang sama dengan app.py
try:
    model = joblib.load('optimized_cervical_cancer_model.pkl')
except FileNotFoundError:
    st.error("File model 'optimized_cervical_cancer_model.pkl' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan aplikasi Anda.")
    st.stop()


# Fungsi untuk melakukan preprocessing dan prediksi
def predict(data):
    # Lakukan prediksi
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    return prediction, prediction_proba

# Judul dan deskripsi aplikasi
st.title('Prediksi Risiko Kanker Serviks')
st.markdown("""
Aplikasi ini menggunakan model *Random Forest* untuk memprediksi risiko kanker serviks (berdasarkan hasil Biopsy) berdasarkan faktor-faktor risiko yang dimasukkan.
**Masukkan data pasien di sidebar untuk melihat hasil prediksi.**
""")

# Input dari pengguna di sidebar
st.sidebar.header('Masukkan Data Pasien')

def user_input_features():
    age = st.sidebar.slider('Usia (Age)', 10, 100, 28)
    num_sexual_partners = st.sidebar.slider('Jumlah Partner Seksual', 1, 30, 3)
    first_sexual_intercourse = st.sidebar.slider('Usia Pertama Kali Berhubungan Seksual', 10, 50, 17)
    num_pregnancies = st.sidebar.slider('Jumlah Kehamilan', 0, 20, 2)
    smokes = st.sidebar.selectbox('Perokok (Smokes)?', ('Tidak', 'Ya'))
    smokes_years = st.sidebar.slider('Lama Merokok (Tahun)', 0, 50, 0)
    smokes_packs_per_year = st.sidebar.slider('Jumlah Pak Rokok per Tahun', 0.0, 100.0, 0.0, 0.1)
    hormonal_contraceptives = st.sidebar.selectbox('Menggunakan Kontrasepsi Hormonal?', ('Tidak', 'Ya'))
    hormonal_contraceptives_years = st.sidebar.slider('Lama Menggunakan Kontrasepsi Hormonal (Tahun)', 0.0, 50.0, 0.0, 0.5)
    iud = st.sidebar.selectbox('Menggunakan IUD?', ('Tidak', 'Ya'))
    iud_years = st.sidebar.slider('Lama Menggunakan IUD (Tahun)', 0.0, 30.0, 0.0, 0.5)
    stds = st.sidebar.selectbox('Pernah Terkena Penyakit Menular Seksual (STDs)?', ('Tidak', 'Ya'))
    stds_number = st.sidebar.slider('Jumlah Jenis STDs', 0, 5, 0)

    # Konversi input ya/tidak ke 0/1
    smokes_val = 1 if smokes == 'Ya' else 0
    hormonal_contraceptives_val = 1 if hormonal_contraceptives == 'Ya' else 0
    iud_val = 1 if iud == 'Ya' else 0
    stds_val = 1 if stds == 'Ya' else 0

    # Membuat dictionary dari input
    data = {
        'Age': age,
        'Number of sexual partners': num_sexual_partners,
        'First sexual intercourse': first_sexual_intercourse,
        'Num of pregnancies': num_pregnancies,
        'Smokes': smokes_val,
        'Smokes (years)': smokes_years,
        'Smokes (packs/year)': smokes_packs_per_year,
        'Hormonal Contraceptives': hormonal_contraceptives_val,
        'Hormonal Contraceptives (years)': hormonal_contraceptives_years,
        'IUD': iud_val,
        'IUD (years)': iud_years,
        'STDs': stds_val,
        'STDs:number': stds_number,
        # Kolom STDs lainnya (diasumsikan 0 jika tidak ada info spesifik)
        'STDs:condylomatosis': 0, 'STDs:cervical condylomatosis': 0, 'STDs:vaginal condylomatosis': 0,
        'STDs:vulvo-perineal condylomatosis': 0, 'STDs:syphilis': 0, 'STDs:pelvic inflammatory disease': 0,
        'STDs:genital herpes': 0, 'STDs:molluscum contagiosum': 0, 'STDs:AIDS': 0, 'STDs:HIV': 0,
        'STDs:Hepatitis B': 0, 'STDs:HPV': 0, 'STDs: Time since first diagnosis': 0, 'STDs: Time since last diagnosis': 0
    }
    
    # Membuat DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Feature Engineering (sama seperti di skrip training)
    features['Age_Group'] = pd.cut(features['Age'], bins=[0, 20, 30, 40, 50, 100], labels=['Teen', '20s', '30s', '40s', '50+'])
    features['First_Sex_Age'] = features['First sexual intercourse']
    features['HPV_Exposure_Duration'] = features['Age'] - features['First sexual intercourse']
    features['Pregnancy_Density'] = features['Num of pregnancies'] / (features['Age'] - 12).clip(lower=1)
    
    return features

# Ambil input dari pengguna
input_df = user_input_features()

# Tampilkan data yang dimasukkan pengguna
st.subheader('Data Pasien yang Dimasukkan')
st.write(input_df)

# Tombol untuk prediksi
if st.button('Lakukan Prediksi'):
    # Lakukan prediksi
    prediction, prediction_proba = predict(input_df)
    
    st.subheader('Hasil Prediksi')
    
    # Tampilkan hasil
    biopsy_status = 'Positif' if prediction[0] == 1 else 'Negatif'
    if biopsy_status == 'Positif':
        st.error(f'Hasil Biopsy: **{biopsy_status}**')
    else:
        st.success(f'Hasil Biopsy: **{biopsy_status}**')

    st.write('Probabilitas Hasil Positif:', prediction_proba[0][1])
    st.write('Probabilitas Hasil Negatif:', prediction_proba[0][0])
    
    st.info('**Disclaimer:** Hasil prediksi ini tidak menggantikan diagnosis medis profesional. Harap konsultasikan dengan dokter untuk evaluasi lebih lanjut.')