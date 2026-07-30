import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Prediksi Harga Cabai Rawit Merah",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("model/model_lr.pkl")
    return model

model = load_model()

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/cabe_rawit_merah_lag1.csv",
        parse_dates=["tanggal_lengkap"]
    )

    df["tanggal_lengkap"] = pd.to_datetime(
        df["tanggal_lengkap"]
    )

    return df


# Dataset awal
df = load_data()


# Dataset aktif
if "dataset" not in st.session_state:
    st.session_state.dataset = df

df = st.session_state.dataset

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

.sticky-nav{
    position: sticky;
    top: 0;
    z-index: 999;
    background: white;
    padding-top: 10px;
    padding-bottom: 10px;
}

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
padding-left:3rem;
padding-right:3rem;
}

.card{

background:#ffffff;

padding:25px;

border-radius:15px;

box-shadow:0px 4px 12px rgba(0,0,0,0.1);

border-left:6px solid #d62828;

margin-bottom:20px;

}

.big-font{

font-size:40px;

font-weight:bold;

color:#d62828;

}

.sub-font{

font-size:18px;

color:#555555;

}

.metric{

background:#fff5f5;

padding:20px;

border-radius:12px;

text-align:center;

border:1px solid #ffd6d6;

}

.info-box{

background:#fff8f8;

padding:15px;

border-radius:10px;

border-left:5px solid red;

}

</style>
""", unsafe_allow_html=True)

# ==========================================
# NAVBAR
# ==========================================

st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=[
        "Home",
        "Dataset",
        "Visualisasi",
        "Prediksi",
        "Tentang"
    ],

    icons=[
        "house-fill",
        "table",
        "bar-chart-fill",
        "graph-up-arrow",
        "person-fill"
    ],

    orientation="horizontal",

    default_index=0
)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# Menu HOME
# ==========================================
if selected == "Home":

    st.markdown("""
    <div class="card">

    <div class="big-font">
    🌶️ Prediksi Harga Cabai Rawit Merah
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📌 Menu Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
    **📂 Dataset**
    
    Melihat data historis harga cabai rawit merah serta mengunggah dataset baru.
    """)
    
    with col2:
        st.info("""
    **📈 Visualisasi**
    
    Melihat perkembangan harga cabai rawit merah dalam bentuk grafik.
    """)
    
    with col3:
        st.info("""
    **🤖 Prediksi**
    
    Melakukan prediksi harga cabai rawit merah berdasarkan model yang telah dibangun.
    """)
    
    st.divider()
    
    st.subheader("ℹ️ Informasi")
    
    st.info("""
    - **Sumber Data** : Open Data Kabupaten Bekasi
    - **Periode Data** : 2023–2025
    
    Dashboard ini dikembangkan sebagai media untuk membantu pengguna melihat
    perkembangan harga cabai rawit merah serta memperoleh prediksi harga berdasarkan
    data historis yang tersedia.
    """)

# ==========================================
# MENU DATASET
# ==========================================
if selected == "Dataset":

    st.title("📊 Dataset")

    st.write("""
    Dataset harga cabai rawit merah periode 2023–2025 di Kabupaten Bekasi.
    """)

    st.divider()

# ==========================================
# DATASET
# ==========================================
    dataset_display = df[["tanggal_lengkap", "cabe_rawit_merah"]].copy()
    dataset_display["tanggal_lengkap"] = dataset_display["tanggal_lengkap"].dt.strftime("%d-%m-%Y")
    dataset_display.columns = ["Tanggal", "Harga"]

    st.dataframe(
        dataset_display,
        use_container_width=True,
        height=500
    )

# ==========================================
# DOWNLOAD DATASET
# ==========================================
    csv=df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset",
        data=csv,
        file_name="dataset_cabe_rawit_merah.csv",
        mime="text/csv"
    )

# ==========================================
# UPLOAD DATASET BARU
# ==========================================

    st.subheader("📂 Upload Dataset Baru")

    st.info("""
    **Ketentuan Dataset**
    Dataset yang diunggah harus memenuhi persyaratan berikut:
    1. Format file **CSV (.csv)**
    2. Memiliki kolom:
        - tanggal_lengkap
        - cabe_rawit_merah
    3. Tidak terdapat nilai kosong (missing value)
    4. Format tanggal menggunakan YYYY-MM-DD
    
    **Catatan**
    Dataset yang diunggah harus sudah melalui proses *feature engineering*.
    Dashboard tidak melakukan pembentukan variabel lag maupun moving average secara otomatis.
    """)

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"]
    )

    required_columns = [
        "tanggal_lengkap",
        "cabe_rawit_merah",
        "lag_1"
    ]

    if uploaded_file is not None:

        with st.spinner("Memproses dataset..."):

            # ==========================
            # Membaca Dataset
            # ==========================

            df_upload = pd.read_csv(
                uploaded_file,
                sep=";"
            )

            # ==========================
            # Validasi Kolom
            # ==========================

            missing_columns = [
                col
                for col in required_columns
                if col not in df_upload.columns
            ]

            if missing_columns:

                st.error(
                    f"❌ Kolom berikut tidak ditemukan: {', '.join(missing_columns)}"
                )

                st.stop()

            # ==========================
            # Validasi Missing Value
            # ==========================

            if df_upload.isnull().values.any():

                st.error(
                    "❌ Dataset masih memiliki nilai kosong (missing value)."
                )

                st.stop()

            # ==========================
            # Validasi Format Tanggal
            # ==========================

            try:

                df_upload["tanggal_lengkap"] = pd.to_datetime(
                    df_upload["tanggal_lengkap"]
                )

            except:

                st.error(
                    "❌ Format kolom tanggal_lengkap tidak valid."
                )

                st.stop()

            # ==========================
            # Urutkan Dataset Upload
            # ==========================

            df_upload = (
                df_upload
                .sort_values("tanggal_lengkap")
                .reset_index(drop=True)
            )

            # ==========================
            # Preview Dataset
            # ==========================

            st.success("✅ Struktur dataset valid.")

            st.subheader("Preview Dataset")

            st.dataframe(
                df_upload,
                use_container_width=True
            )

            # ==========================
            # Informasi Dataset
            # ==========================

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Jumlah Data",
                    len(df_upload)
                )

            with col2:
                st.metric(
                    "Jumlah Kolom",
                    len(df_upload.columns)
                )

            with col3:
                st.metric(
                    "Tanggal Awal",
                    df_upload["tanggal_lengkap"].min().strftime("%d-%m-%Y")
                )

            with col4:
                st.metric(
                    "Tanggal Akhir",
                    df_upload["tanggal_lengkap"].max().strftime("%d-%m-%Y")
                )

            # ==========================
            # Gunakan Dataset
            # ==========================

            if st.button("🚀 Gunakan Dataset"):

                dataset_lama = st.session_state.dataset

                dataset_lama["tanggal_lengkap"] = pd.to_datetime(
                    dataset_lama["tanggal_lengkap"]
                )

                dataset_baru = pd.concat(
                    [
                        dataset_lama,
                        df_upload
                    ],
                    ignore_index=True
                )

                dataset_baru = (
                    dataset_baru
                    .drop_duplicates(
                        subset="tanggal_lengkap",
                        keep="last"
                    )
                    .sort_values("tanggal_lengkap")
                    .reset_index(drop=True)
                )

                st.session_state.dataset = dataset_baru

                st.success(
                    "✅ Dataset berhasil diperbarui."
                )

                st.info(f"""
    Jumlah Data Aktif : **{len(dataset_baru)}**

    Periode Dataset :

    **{dataset_baru['tanggal_lengkap'].min().strftime('%d-%m-%Y')}**
    s.d.
    **{dataset_baru['tanggal_lengkap'].max().strftime('%d-%m-%Y')}**

    Silakan buka menu **Prediksi**.
    Periode tanggal akan otomatis mengikuti dataset terbaru.
    """)

# ==========================================
# MENU VISUALISASI
# ==========================================

if selected=="Visualisasi":

    st.title("📈 Visualisasi Dataset")

    st.write("""
    Grafik berikut menunjukkan perkembangan harga cabai rawit merah berdasarkan data historis periode 2023–2025.
    """)

    st.divider()

# ==========================================
# GRAFIK HARGA AKTUAL
# ==========================================
    st.subheader("Harga Cabai Rawit Merah")

    fig,ax=plt.subplots(figsize=(12,4))

    ax.plot(

        df["tanggal_lengkap"],

        df["cabe_rawit_merah"],

        linewidth=2

    )

    ax.set_xlabel("Tanggal")

    ax.set_ylabel("Harga")

    ax.grid(True)

    st.pyplot(fig)

# ==========================================
# RECURSIVE FORECAST
# ==========================================

def recursive_forecast(model, df, target_date, harga_awal = None):

    df_forecast = df.copy().sort_values("tanggal_lengkap").reset_index(drop=True)

    current_date = df_forecast.iloc[-1]["tanggal_lengkap"]
    if harga_awal is not None:
        df_forecast.loc[
            df_forecast.index[-1],
            "cabe_rawit_merah"
        ] = harga_awal
    while current_date.date() < target_date:

        last = df_forecast.iloc[-1]

        next_date = current_date + pd.Timedelta(days=1)

        hari_ke = last["hari_ke"] + 1

        # ==========================
        # Membentuk variabel lag
        # ==========================

        lag1 = df_forecast.iloc[-1]["cabe_rawit_merah"]

        # ==========================
        # Prediksi
        # ==========================

        X = np.array([[

            lag1,

        ]])

        prediksi = model.predict(X)[0]

        # ==========================
        # Simpan hasil prediksi
        # ==========================

        row = pd.DataFrame({

            "tanggal_lengkap":[next_date],

            "cabe_rawit_merah":[prediksi],

            "lag_1":[lag1],

        })

        df_forecast = pd.concat(

            [

                df_forecast,

                row

            ],

            ignore_index=True

        )

        current_date = next_date

    return df_forecast

# ==========================================
# MENU PREDIKSI
# ==========================================

if selected == "Prediksi":

    df = st.session_state.dataset.copy()

    df["tanggal_lengkap"] = pd.to_datetime(
        df["tanggal_lengkap"]
    )

    min_date = df["tanggal_lengkap"].min().date()
    max_date = df["tanggal_lengkap"].max().date()

    st.title("🔮 Prediksi Harga Cabai Rawit Merah")

    st.write("""
    Pilih tanggal untuk melakukan prediksi harga cabai rawit merah.
    """)

    st.divider()

    tanggal = st.date_input(
        "📅 Pilih Tanggal",
        value=max_date,
        min_value=min_date
    )

    # ==========================================
    # PREDIKSI HISTORIS
    # ==========================================
    
    if tanggal <= max_date:
    
        data = df[df["tanggal_lengkap"].dt.date == tanggal]
    
        if data.empty:
            st.error("Data untuk tanggal tersebut tidak ditemukan.")
            st.stop()
    
        data = data.iloc[0]
    
        lag1 = data["lag_1"]
        aktual = data["cabe_rawit_merah"]
    
        if st.button("🚀 Prediksi Harga"):
    
            with st.spinner("Sedang melakukan prediksi..."):
    
                X = np.array([[
        
                    lag1,
            
                ]])
    
                prediksi = model.predict(X)[0]
    
            st.success("Prediksi berhasil dilakukan.")
    
            col1, col2, col3 = st.columns(3)
    
            with col1:
                st.metric(
                    "Harga Aktual",
                    f"Rp {aktual:,.0f}"
                )
    
            with col2:
                st.metric(
                    "Harga Prediksi",
                    f"Rp {prediksi:,.0f}"
                )
    
            with col3:
                selisih = prediksi - aktual
    
                st.metric(
                    "Selisih",
                    f"{selisih:,.0f}"
                )
    
            st.info("""
    
    **Catatan**
    
    Prediksi dilakukan menggunakan data historis yang tersedia pada dataset.
    Hasil prediksi dapat dibandingkan langsung dengan nilai aktual sehingga
    dapat diketahui besarnya selisih prediksi.
    
    """)
    
    # ==========================================
    # RECURSIVE FORECAST
    # ==========================================
    
    else:
        mode = st.radio(
            "Sumber Data Awal",
            [
                "Otomatis dari Dataset",
                "Input Manual"
            ]
        )

        harga_manual = None

        if mode == "Input Manual":

            harga_manual = st.number_input(
                "Masukkan Harga Hari Sebelumnya (Lag-1)",
                min_value=0.0,
                step=100.0,
                format="%.0f"
            )
    
        if st.button("🚀 Prediksi Harga"):
    
            with st.spinner("Melakukan recursive forecasting..."):
    
                hasil = recursive_forecast(
                    model,
                    df,
                    tanggal,
                    harga_awal=harga_manual
                )
    
            st.success("Prediksi berhasil dilakukan.")
    
            hasil_prediksi = hasil.iloc[-1]
    
            # ==========================
            # Hasil Prediksi
            # ==========================
    
            col1, col2 = st.columns(2)
    
            with col1:
    
                st.metric(
                    "Tanggal Prediksi",
                    tanggal.strftime("%d-%m-%Y")
                )
    
            with col2:
    
                st.metric(
                    "Harga Prediksi",
                    f"Rp {hasil_prediksi['cabe_rawit_merah']:,.0f}"
                )
    
            st.info("""
    
    **Catatan**
    
    Prediksi dilakukan menggunakan data historis yang tersedia pada dataset. Hasil prediksi
    setelah periode tersebut merupakan estimasi berdasarkan pola historis dan
    dapat berbeda dengan harga aktual.
    
    """)
    
# ==========================================
# MENU TENTANG
# ==========================================

if selected == "Tentang":

    st.title("ℹ️ Tentang")

    st.subheader("👩‍🎓 Profil Peneliti")

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(
            "assets/foto_raina.jpg",
            width=220
        )

    with col2:

        st.write("**Nama**")
        st.write("Raina Andriani Putri")

        st.write("**NPM**")
        st.write("202210715283")

        st.write("**Instansi**")
        st.write("Universitas Bhayangkara Jakarta Raya")

        st.write("**Program Studi**")
        st.write("Informatika")

    st.subheader("📄 Judul Skripsi")

    st.success("""

Prediksi Harga Cabai Rawit Merah
Menggunakan Metode
Linear Regression

""")
    
    st.subheader("🧠 Metode")

    st.write("""

Model dibangun menggunakan metode
**Linear Regression**
dengan variabel:

- hari_ke
- lag-1

""")

    st.subheader("📊 Dataset")

    st.write("""

Sumber data: https://opendata.bekasikab.go.id/

Periode: 2023–2025

""")

    st.subheader("💻 Teknologi")

    st.write("""

Dashboard dibuat menggunakan:
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

""")
    
st.divider()

st.markdown(
    """
    <center>

    © 2026    
    Universitas Bhayangkara Jakarta Raya

    </center>
    """,
    unsafe_allow_html=True
)
