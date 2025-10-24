import streamlit as st
import tensorflow as tf
import numpy as np

def load_model():
    """
    Загрузка модели
    """
    model = tf.keras.models.load_model('brain_tumor_1_model.keras')
    return model

def load_image(file_bytes):
    """
    Загрузка данных
    """
    image = tf.image.decode_image(file_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (32, 32))
    # grayscaling
    image = tf.image.rgb_to_grayscale(image)
    # normalize
    image = tf.cast(image, tf.float32) / 255.0
    # reshape
    image = tf.reshape(image, shape=(1,32,32,1))
    return image

def make_prediction(image):
    """
    Использование модели
    """
    prediction = model.predict(image, verbose=2)
    class_deseas = class_names[np.argmax(prediction)]
    return class_deseas

# Загружаем предварительно обученную модель
model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Выводим заголовок страницы
st.title("Brain tumor deseas")

# Простой загрузчик файлов
uploaded_file = st.file_uploader(
    "Нажмите чтобы выбрать изображение",
    type=['jpg', 'jpeg', 'png', 'gif']
)

if uploaded_file is not None:
    # Загружаем изображение
    file_bytes = uploaded_file.getvalue()
    image = load_image(file_bytes)
    class_deseas = make_prediction(image)
    # Показываем название файла как label
    st.write(f"**Выбранное изображение:** {class_deseas}")
    # Отображаем изображение
    st.image(uploaded_file)
else:
    st.write("Изображение не выбрано")
