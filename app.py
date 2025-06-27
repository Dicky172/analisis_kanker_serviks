import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Risiko Kanker Serviks",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_resource
def load_pipeline(filepath: str):
    """
    Memuat pipeline preprocessing dan model dari file .pkl.
    Menggunakan cache untuk efisiensi agar tidak loading berulang kali.
    """
    try:
        pipeline = joblib.load(filepath)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: File model '{filepath}' tidak ditemukan.")
        st.error("Pastikan file .pkl berada di root direktori repositori Anda.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

def perform_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan feature engineering yang sama persis dengan skrip training.
    """
    # Buat salinan untuk menghindari SettingWithCopyWarning
    df_engineered = df.copy()
    
    # 1. Kelompokkan usia
    df_engineered['Age_Group'] = pd.cut(
        df_engineered['Age'],
        bins=[0, 20, 30, 40, 50, 100],
        labels=['Teen', '20s', '30s', '40s', '50+']
    )
    
    # 2. Buat fitur baru (sesuai skrip training)
    df_engineered['First_Sex_Age'] = df_engineered['First sexual intercourse']
    df_engineered['HPV_Exposure_Duration'] = df_engineered['Age'] - df_engineered['First sexual intercourse']
    # .clip(lower=1) untuk menghindari pembagian dengan nol
    df_engineered['Pregnancy_Density'] = df_engineered['Num of pregnancies'] / (df_engineered['Age'] - 12).clip(lower=1)
    
    return df_engineered

# --- UI APLIKASI ---

def main():
    """
    Fungsi utama untuk menjalankan aplikasi Streamlit.
    """
    # Nama file pkl yang dihasilkan dari skrip training Anda
    PIPELINE_FILEPATH = 'optimized_cervical_cancer_model.pkl'
    
    # Memuat pipeline
    pipeline = load_pipeline(PIPELINE_FILEPATH)

    # Tampilan Judul dan Deskripsi
    st.title("🩺 Aplikasi Prediksi Faktor Risiko Kanker Serviks")
    st.markdown("""
    Aplikasi ini menggunakan model **Random Forest** yang telah dioptimalkan untuk memberikan estimasi 
    risiko kanker serviks. Masukkan data pasien di panel sebelah kiri untuk melihat hasil prediksi.
    """)
    st.warning("**Disclaimer:** Aplikasi ini adalah alat bantu dan **bukan pengganti diagnosis medis profesional**. Selalu konsultasikan dengan dokter untuk evaluasi kesehatan yang akurat.", icon="⚠️")
    st.divider()

    # Hanya jalankan jika pipeline berhasil dimuat
    if pipeline is not None:
        st.sidebar.header("📝 Masukkan Data Pasien")
        
        # Daftar fitur input yang dibutuhkan oleh model (sesuai 'X' di skrip training)
        # Urutan ini penting untuk di-debug, meskipun pipeline akan menanganinya
        input_feature_list = [
            'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 
            'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 
            'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis', 'STDs:cervical condylomatosis', 
            'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 
            'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 
            'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 'STDs: Time since first diagnosis', 
            'STDs: Time since last diagnosis'
        ]
        
        user_input = {}
        with st.sidebar.form(key='patient_data_form'):
            for feature in input_feature_list:
                # Format label agar lebih mudah dibaca
                label = feature.replace('_', ' ').replace(':', ' - ').title()
                user_input[feature] = st.number_input(
                    label=label,
                    key=feature,
                    value=0.0,
                    step=1.0, # Sesuaikan step jika perlu
                    format="%.2f"
                )
            
            submit_button = st.form_submit_button(label="SUBMIT & PREDIKSI", type="primary")

        # Proses saat tombol ditekan
        if submit_button:
            # 1. Konversi input ke DataFrame
            input_df = pd.DataFrame([user_input])
            
            # 2. Lakukan feature engineering
            engineered_df = perform_feature_engineering(input_df)
            
            # 3. Prediksi menggunakan pipeline
            # Pipeline akan secara otomatis melakukan scaling dan one-hot encoding
            try:
                prediction = pipeline.predict(engineered_df)
                prediction_proba = pipeline.predict_proba(engineered_df)

                # 4. Tampilkan hasil
                st.header("🔬 Hasil Prediksi")
                prob_negatif, prob_positif = prediction_proba[0]
                
                col1, col2 = st.columns([2,3])
                with col1:
                    if prediction[0] == 1:
                        st.error("**Risiko Tinggi (Positif)**", icon="🚨")
                    else:
                        st.success("**Risiko Rendah (Negatif)**", icon="✅")
                
                with col2:
                    if prediction[0] == 1:
                        st.metric(label="Tingkat Keyakinan Prediksi (Risiko Tinggi)", value=f"{prob_positif*100:.2f}%")
                    else:
                        st.metric(label="Tingkat Keyakinan Prediksi (Risiko Rendah)", value=f"{prob_negatif*100:.2f}%")

                st.write("""
                Berdasarkan data yang dimasukkan, model memberikan estimasi seperti di atas. 
                Probabilitas yang tinggi menunjukkan keyakinan model terhadap prediksinya.
                """)
                
                with st.expander("Lihat Detail Probabilitas"):
                    st.json({
                        "Probabilitas Risiko Rendah (Kelas 0)": f"{prob_negatif:.4f}",
                        "Probabilitas Risiko Tinggi (Kelas 1)": f"{prob_positif:.4f}"
                    })

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

        else:
            st.info("Silakan isi data pasien pada panel di sebelah kiri, lalu klik tombol 'SUBMIT & PREDIKSI' untuk melihat hasilnya.", icon="ℹ️")
    else:
        st.error("Aplikasi tidak dapat dijalankan karena gagal memuat file model. Silakan periksa kembali file `optimized_cervical_cancer_model.pkl` Anda.")


if __name__ == '__main__':
    main()
