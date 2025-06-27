import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Fungsi untuk Memuat Aset Model ---
# Menggunakan cache agar model tidak di-load ulang setiap kali ada interaksi.
@st.cache_resource
def load_model_assets():
    """Memuat model, scaler, dan nama fitur yang telah disimpan."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error(
            "File model/scaler/fitur tidak ditemukan. "
            "Pastikan file 'best_model.pkl', 'scaler.pkl', dan 'feature_names.pkl' "
            "berada di folder yang sama dengan 'app.py'."
        )
        return None, None, None

# Memuat aset di awal
model, scaler, feature_names = load_model_assets()

# --- Tampilan Aplikasi Web (UI) ---
st.set_page_config(page_title="Prediksi Kanker Serviks", layout="wide")

st.title("🩺 Aplikasi Prediksi Faktor Risiko Kanker Serviks")

st.write("""
Aplikasi ini menggunakan model Machine Learning (Random Forest) untuk memprediksi 
apakah seorang pasien berisiko tinggi terkena kanker serviks berdasarkan data klinis dan demografis.
""")
st.warning("**Disclaimer:** Hasil prediksi ini bukan merupakan diagnosis medis. Selalu konsultasikan dengan dokter atau tenaga medis profesional untuk diagnosis yang akurat.", icon="⚠️")

st.divider()

# Hanya tampilkan form input jika model berhasil dimuat
if model and scaler and feature_names:
    st.header("Masukkan Data Pasien:")

    # Membuat kolom untuk layout yang lebih rapi
    col1, col2, col3 = st.columns(3)

    # Dictionary untuk menampung input pengguna
    user_input = {}
    
    # Membagi fitur ke dalam 3 kolom secara dinamis
    features_per_col = int(np.ceil(len(feature_names) / 3))
    
    # Fungsi untuk membuat input field
    def create_input_field(feature_name):
        # Menggunakan session_state untuk menyimpan nilai default atau nilai terakhir yang dimasukkan
        if feature_name not in st.session_state:
            st.session_state[feature_name] = 0.0
        return st.number_input(
            label=feature_name.replace('_', ' ').capitalize(), 
            key=feature_name,
            format="%.2f"
        )

    with col1:
        for feature in feature_names[0:features_per_col]:
            user_input[feature] = create_input_field(feature)

    with col2:
        for feature in feature_names[features_per_col : 2*features_per_col]:
            user_input[feature] = create_input_field(feature)
    
    with col3:
        for feature in feature_names[2*features_per_col:]:
            user_input[feature] = create_input_field(feature)

    st.divider()

    # Tombol untuk melakukan prediksi
    if st.button("SUBMIT & PREDIKSI", type="primary", use_container_width=True):
        try:
            # --- BAGIAN KRUSIAL YANG DIPERBAIKI ---
            # 1. Buat DataFrame dari input pengguna.
            input_df = pd.DataFrame([user_input])

            # 2. Pastikan urutan kolom 100% sama dengan saat training.
            # Ini adalah langkah paling penting untuk menghindari KeyError.
            input_df = input_df[feature_names]
            
            # 3. Lakukan penskalaan pada data input menggunakan scaler yang sudah di-load.
            input_scaled = scaler.transform(input_df)

            # 4. Lakukan prediksi.
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            # 5. Tampilkan hasil prediksi.
            st.header("Hasil Prediksi:")
            prob_negatif, prob_positif = prediction_proba[0]

            if prediction[0] == 1:
                st.error(f"**Risiko Tinggi (Positif)**", icon="🚨")
                st.write(f"Model memprediksi pasien ini memiliki risiko tinggi untuk kanker serviks dengan probabilitas **{prob_positif*100:.2f}%**.")
            else:
                st.success(f"**Risiko Rendah (Negatif)**", icon="✅")
