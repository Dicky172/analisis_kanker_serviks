import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# --- KONFIGURASI APLIKASI ---
# Mengatur konfigurasi halaman. Ini harus menjadi perintah pertama Streamlit.
st.set_page_config(
    page_title="Prediksi Risiko Kanker Serviks",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_resource
def load_deployment_assets(filepath: str):
    """
    Memuat semua aset (model, scaler, nama fitur) dari satu file .pkl.
    Menggunakan cache untuk performa yang lebih baik agar file tidak dimuat ulang.
    """
    try:
        assets = joblib.load(filepath)
        model = assets['model']
        scaler = assets['scaler']
        feature_names = assets['features']
        # Validasi sederhana untuk memastikan semua aset ada
        if not all([isinstance(model, BaseEstimator), 
                    hasattr(scaler, 'transform'), 
                    isinstance(feature_names, list)]):
            raise ValueError("File aset tidak valid atau isinya tidak lengkap.")
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error(f"Error: File aset '{filepath}' tidak ditemukan. Pastikan file ini ada di root repositori GitHub Anda.")
        return None, None, None
    except (KeyError, ValueError) as e:
        st.error(f"Error saat membaca file aset: {e}. Pastikan file .pkl dibuat dengan benar dan berisi keys 'model', 'scaler', dan 'features'.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan yang tidak terduga saat memuat aset: {e}")
        return None, None, None

def create_input_form(feature_names: list) -> dict:
    """
    Membuat form input untuk data pasien di sidebar Streamlit.
    """
    st.sidebar.header("📝 Masukkan Data Pasien")
    user_input = {}
    
    # Menggunakan st.form untuk mengelompokkan input
    with st.sidebar.form(key='patient_data_form'):
        for feature in feature_names:
            # Mengganti '_' dengan spasi dan membuat judul lebih rapi
            label = feature.replace('_', ' ').title()
            user_input[feature] = st.number_input(
                label=label,
                key=feature,
                value=0.0,
                step=0.1,
                format="%.2f"
            )
        
        # Tombol submit untuk form
        submit_button = st.form_submit_button(label="SUBMIT & PREDIKSI", type="primary")
            
    return user_input, submit_button

def predict_risk(model: BaseEstimator, scaler: TransformerMixin, feature_names: list, input_data: dict) -> tuple:
    """
    Melakukan penskalaan dan prediksi pada data input.
    """
    try:
        # Mengonversi input dictionary ke DataFrame dengan urutan kolom yang benar
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # Melakukan penskalaan
        scaled_data = scaler.transform(input_df)
        
        # Melakukan prediksi
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Terjadi error saat proses prediksi: {e}")
        return None, None

def display_results(prediction: np.ndarray, prediction_proba: np.ndarray):
    """
    Menampilkan hasil prediksi kepada pengguna dengan format yang jelas.
    """
    st.header("🔬 Hasil Prediksi")
    prob_negatif, prob_positif = prediction_proba[0]

    # Menggunakan kolom untuk tata letak yang lebih baik
    col1, col2 = st.columns(2)

    if prediction[0] == 1:
        with col1:
            st.error("**Risiko Tinggi (Positif)**", icon="🚨")
        with col2:
            st.metric(label="Tingkat Keyakinan Prediksi", value=f"{prob_positif*100:.2f}%")
        st.write("Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien ini memiliki **risiko tinggi** untuk kanker serviks.")
    else:
        with col1:
            st.success("**Risiko Rendah (Negatif)**", icon="✅")
        with col2:
            st.metric(label="Tingkat Keyakinan Prediksi", value=f"{prob_negatif*100:.2f}%")
        st.write("Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien ini memiliki **risiko rendah** untuk kanker serviks.")

    # Menampilkan detail probabilitas dalam expander
    with st.expander("Lihat Detail Probabilitas"):
        st.json({
            "Probabilitas Risiko Rendah (Kelas 0)": f"{prob_negatif:.4f}",
            "Probabilitas Risiko Tinggi (Kelas 1)": f"{prob_positif:.4f}"
        })

# --- FUNGSI UTAMA APLIKASI ---

def main():
    """
    Fungsi utama untuk menjalankan alur aplikasi Streamlit.
    """
    # Ganti dengan nama file .pkl Anda
    ASSET_FILEPATH = 'optimized_cervical_cancer_model.pkl'
    
    st.title("Aplikasi Prediksi Faktor Risiko Kanker Serviks")
    st.markdown("""
    Aplikasi ini menggunakan model *Machine Learning* untuk memberikan estimasi risiko kanker serviks 
    berdasarkan faktor-faktor klinis dan demografis.
    """)
    st.warning("**Disclaimer:** Ini bukan diagnosis medis. Hasil prediksi hanyalah indikasi awal dan tidak dapat menggantikan konsultasi profesional dengan dokter atau ahli medis.", icon="⚠️")
    
    # Memuat aset
    model, scaler, feature_names = load_deployment_assets(ASSET_FILEPATH)

    # Hanya lanjutkan jika aset berhasil dimuat
    if all([model, scaler, feature_names]):
        user_input, submit_button = create_input_form(feature_names)
        
        if submit_button:
            prediction, prediction_proba = predict_risk(model, scaler, feature_names, user_input)
            
            # Tampilkan hasil jika prediksi berhasil
            if prediction is not None:
                display_results(prediction, prediction_proba)
        else:
            # Pesan awal di halaman utama
            st.info("Silakan isi data pasien pada panel di sebelah kiri dan klik tombol 'SUBMIT & PREDIKSI' untuk melihat hasilnya.", icon="ℹ️")
    else:
        st.error("Aplikasi tidak dapat berjalan karena gagal memuat aset model. Harap periksa kembali file aset dan pesan error di atas.")

if __name__ == '__main__':
    main()
