import streamlit as st
import requests
from PIL import Image
import numpy as np
import pandas as pd


st.set_page_config(
    page_title="Stingless Bee Classifier",
    page_icon="🐝",
    layout="centered"
)

# CSS

st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }

    h1.centered {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
        font-size: clamp(22px, 4.5vw, 40px);
        line-height: 1.1;
        margin-top: -1em;
        margin-bottom: 1em;
    }

    h3.centered {
        font-size: clamp(14px, 2.2vw, 22px);
        margin-top: -2.5em;
        margin-bottom: 5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Helper Format persentase

def format_percent(value: float) -> str:
    val = round(value, 2)
    if val.is_integer():
        return f"{int(val)}%"
    elif (val * 10).is_integer():
        return f"{val:.1f}%".replace(".", ",")
    else:
        return f"{val:.2f}%".replace(".", ",")


# API Endpoint

API_URL = "http://localhost:5000/predict"


st.markdown("<h1 class='centered'>🐝</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='centered'>Klasifikasi Lebah Madu Tanpa Sengat</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='centered'>Berbasis Inception ResNet V2</h3>", unsafe_allow_html=True)


# Upload Gambar

uploaded_file = st.file_uploader(
    "📤 Unggah gambar untuk melakukan klasifikasi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Gambar Terunggah", use_container_width=True)

    if st.button("🔍 Klasifikasikan"):
        with st.spinner("Memproses gambar..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    label = result.get("predicted_label", "Tidak tersedia")
                    confidence = result.get("confidence", 0.0)
                    is_ood = result.get("is_ood", False)

                    st.markdown("---")
                    st.subheader("🧠 Hasil Klasifikasi")

                    # Status Prediksi
                    
                    if is_ood:
                        st.error("🚫 Objek tidak dikenali")
                    else:
                        col1, col2 = st.columns(2)
                        col1.metric("Spesies Lebah", label.replace("_", " "))
                        col2.metric("Akurasi Prediksi", format_percent(confidence))

                    
                    # OUTPUT MENTAH
                    
                    with st.expander("📊 Detail Hasil"):
                        logits = np.array(result.get("raw_logits", []))

                        if logits.size > 0:
                            logits = logits[0]
                            probs = np.exp(logits) / np.sum(np.exp(logits))

                            class_names = [
                                "Heterotrigona itama",
                                "Tetrigona apicalis",
                                "Tetrigona binghami",
                                "Tetrigona vidua"
                            ]

                            df = pd.DataFrame({
                                "Spesies": class_names,
                                "Probabilitas (%)": [format_percent(p * 100) for p in probs]
                            })

                            st.table(df)

                            if is_ood:
                                st.warning(
                                    "⚠️ Model belum cukup yakin untuk mengenali objek pada gambar ini sehingga objek diklasifikasikan sebagai tidak dikenali. Coba gunakan gambar lain."
                                )
                        else:
                            st.info("Logits tidak tersedia dari API.")

                else:
                    st.error(f"❌ Gagal memproses gambar: {response.text}")

            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {str(e)}")
