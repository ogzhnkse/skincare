import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- AYARLAR ---
# Model dosyasının adı (yan yana olduklarından emin olun)
MODEL_PATH = 'cilt_kanseri_modeli.h5'

# Sınıf etiketleri (Eğitimdeki alfabetik sıraya göre: 0=Benign, 1=Malignant)
LABELS = {0: 'Zararsız (Benign)', 1: 'RİSKLİ (Malignant)'}


# --- 1. MODELİ YÜKLE ---
# @st.cache_resource sayesinde model her defasında tekrar yüklenmez, hız kazanır.
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# Uygulama açılırken modeli yükle
with st.spinner('Yapay Zeka Modeli Yükleniyor...'):
    model = load_model()

# --- 2. ARAYÜZ TASARIMI ---
st.title("LUMORA")
st.markdown("""
Bu uygulama, yüklediğiniz cilt lezyonu fotoğraflarını yapay zeka ile analiz eder.
**UYARI:** *Bu sonuçlar sadece bir tahmindir ve tıbbi teşhis yerine geçmez. Kesin sonuç için doktora başvurun.*
""")

# Fotoğraf yükleme seçenekleri (Hem dosya seçme hem kamera)
option = st.radio("Fotoğrafı nasıl yüklemek istersiniz?", ("Dosya Yükle", "Kamera Kullan"))

image = None

if option == "Dosya Yükle":
    uploaded_file = st.file_uploader("Bir resim dosyası seçin", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Fotoğraf Çek")
    if camera_file is not None:
        image = Image.open(camera_file)

# --- 3. TAHMİN İŞLEMİ ---
if image is not None:
    # Kullanıcıya resmi göster
    st.image(image, caption='Analiz Edilen Görüntü', use_column_width=True)

    # Resmi modelin anlayacağı formata getir (Preprocessing)
# 1. Önce görüntünün tam ortasını kırpalım (Zoom etkisi)
    # Görüntü genişliğinin %50'sini al, kenarları at.
    width, height = image.size
    new_width = width * 0.5 
    new_height = height * 0.5
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    image = image.crop((left, top, right, bottom))

    # 2. Sonra modele uygun boyuta getir
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)

    # 2. Sayısal diziye çevir ve normalize et (0-1 arası)
    img_array = np.array(image) / 255.0

    # 3. Boyut ekle (Model (1, 224, 224, 3) şeklinde ister)
    img_reshaped = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    prediction = model.predict(img_reshaped)

    # Sonuçları al
    # prediction[0][0] -> Benign olasılığı
    # prediction[0][1] -> Malignant olasılığı
    malignant_prob = prediction[0][1]
    benign_prob = prediction[0][0]

    st.write("---")
    st.subheader("📊 Analiz Sonucu")

    # Sonucu ekrana yazdır
    if malignant_prob > 0.5:
        st.error(f"⚠️ SONUÇ: **{LABELS[1]}**")
        st.write(f"Risk Oranı: **%{malignant_prob * 100:.2f}**")
        st.info("Lütfen uzman bir dermatoloğa görünün.")
    else:
        st.success(f"✅ SONUÇ: **{LABELS[0]}**")
        st.write(f"Zararsız Olma İhtimali: **%{benign_prob * 100:.2f}**")
