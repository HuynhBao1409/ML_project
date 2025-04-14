import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Tải mô hình đã được huấn luyện sẵn
model = load_model('cnn_model.h5')

# Định nghĩa nhãn các lớp (thay thế bằng nhãn lớp thực tế của bạn)
class_labels = {0: 'táo', 1: 'chuối', 2: 'củ dền', 3: 'ớt chuông', 4: 'bắp cải', 5: 'ớt', 6: 'cà rốt',
          7: 'súp lơ', 8: 'ớt', 9: 'ngô', 10: 'dưa chuột', 11: 'cà tím', 12: 'tỏi', 13: 'gừng',
          14: 'nho', 15: 'ớt jalapeno', 16: 'kiwi', 17: 'chanh', 18: 'rau xà lách',
          19: 'xoài', 20: 'hành tây', 21: 'cam', 22: 'ớt paprika', 23: 'lê', 24: 'đậu Hà Lan', 25: 'dứa',
          26: 'lựu', 27: 'khoai tây', 28: 'củ cải', 29: 'đậu nành', 30: 'rau chân vịt', 31: 'ngô ngọt',
          32: 'khoai lang', 33: 'cà chua', 34: 'củ cải trắng', 35: 'dưa hấu'} 

# Hàm tiền xử lý hình ảnh
def preprocess_image(image):
    img = image.resize((224, 224))  # Thay đổi kích thước theo đầu vào mong đợi của mô hình
    img = np.array(img)
    #img = img / 255.0  # Chuẩn hóa hình ảnh
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    return img

# Hàm dự đoán
def predict(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels[predicted_class[0]]

# Ứng dụng Streamlit
st.image("banner.png", use_container_width=True)
st.title("Phân Loại Trái Cây và Rau Củ")

uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    label = predict(image)
    st.markdown(f"<h2 style='color: green;'>Dự đoán: {label}</h2>", unsafe_allow_html=True)
    st.image(image, caption='Hình ảnh đã tải lên.', use_container_width=True)
    st.write("")