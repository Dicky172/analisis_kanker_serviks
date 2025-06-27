import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

# --- UI UTAMA ---
st.set_page_config(page_title="Prediksi Kanker Serviks", layout="wide")
st.title("🩺 Aplikasi Prediksi Faktor Risiko Kanker Serviks")

st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko kanker serviks.")
st.warning("**Disclaimer:** Ini bukan diagnosis medis. Selalu konsultasikan dengan dokter.", icon="⚠️")
st.divider()

# Memuat aset di awal
model, scaler, feature_names = load_model_assets()

# Hanya tampilkan form input jika model berhasil dimuat
if not all([model, scaler, feature_names]):
    st.warning("Aplikasi tidak dapat berjalan karena file model tidak dapat dimuat.")
else:
    st.header("Masukkan Data Pasien:")

    # Kolom untuk layout
    col1, col2, col3 = st.columns(3)
    
    # Dictionary untuk menampung input
    user_input = {}
    
    # Membagi fitur ke dalam 3 kolom
    features_per_col = int(np.ceil(len(feature_names) / 3))
    
    def create_input_field(feature_name):
        return st.number_input(
            label=feature_name.replace('_', ' ').capitalize(), 
            key=feature_name,
            value=0.0,
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

    # Tombol prediksi
    if st.button("SUBMIT & PREDIKSI", type="primary", use_container_width=True):
        # Blok try-except untuk menangani error saat prediksi
        try:
            # 1. Buat DataFrame dari input dan pastikan urutan kolom benar
            input_df = pd.DataFrame([user_input])[feature_names]
            
            # 2. Skalakan data input
            input_scaled = scaler.transform(input_df)

            # 3. Lakukan prediksi
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            # 4. Tampilkan hasil
            st.header("Hasil Prediksi:")
            prob_negatif, prob_positif = prediction_proba[0]

            if prediction[0] == 1:
                st.error(f"**Risiko Tinggi (Positif)**", icon="🚨")
                st.write(f"Model memprediksi pasien ini memiliki risiko tinggi untuk kanker serviks dengan probabilitas **{prob_positif*100:.2f}%**.")
            else:
                st.success(f"**Risiko Rendah (Negatif)**", icon="✅")
                st.write(f"Model memprediksi pasien ini memiliki risiko rendah untuk kanker serviks. Probabilitas hasil negatif adalah **{prob_negatif*100:.2f}%**.")

            st.write("Detail Probabilitas:")
            st.json({
                "Probabilitas Hasil Negatif (Kelas 0)": f"{prob_negatif:.4f}",
                "Probabilitas Hasil Positif (Kelas 1)": f"{prob_positif:.4f}"
            })

        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")
            st.error("Ini kemungkinan disebabkan oleh masalah pada file model atau scaler. Coba latih ulang model dan simpan kembali file .pkl.")
