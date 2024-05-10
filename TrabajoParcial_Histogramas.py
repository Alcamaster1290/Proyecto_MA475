import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import plotly.express as px

# Función para mostrar el histograma de una imagen
def plot_histogram(image):
    # Convertir la imagen PIL a un array numpy
    img_array = np.array(image)

    # Calcular el histograma de la imagen
    histogram = np.histogram(img_array, bins=256, range=(0, 256))[0]

    # Crear el gráfico de barras con Plotly
    fig = px.bar(x=np.arange(256), y=histogram, labels={'x':'Niveles de Gris', 'y':'Frecuencia'},
                 title='Histograma de Niveles de Gris', width=600, height=400)
    
    # Mostrar el gráfico
    st.plotly_chart(fig)
    if st.button('Guardar Histograma'):
        st.download_button('histograma.png',fig)


# Función para preprocesar la imagen: ajusta el tamaño y la convierte a escala de grises
def preprocess_image(image):
    # Ajustar tamaño a una cantidad fija de píxeles
    target_width = 1000
    scale_factor = target_width / image.width
    resized_image = image.resize((target_width, int(image.height * scale_factor)))

    # Convertir a escala de grises
    gray_image = resized_image.convert('L')

    # Mostrar información sobre la imagen
    st.write(f'La imagen tiene {gray_image.width} píxeles de ancho y {gray_image.height} píxeles de alto')
    
    return gray_image


# Función para expandir el histograma de una imagen
def expand_histogram(image):
    # Obtener el histograma de la imagen original
    original_hist, bins = np.histogram(np.array(image).ravel(), bins=256, range=(0,256))

    # Encontrar los valores mínimo y máximo de intensidad
    min_val, max_val = np.min(image), np.max(image)

    # Escalar linealmente los valores de píxeles
    expanded_image = (image - min_val) * (255 / (max_val - min_val))
    
    # Convertir a formato de imagen
    expanded_image = Image.fromarray(expanded_image.astype(np.uint8))

    # Obtener el histograma de la imagen expandida
    expanded_hist, bins = np.histogram(np.array(expanded_image).ravel(), bins=256, range=(0,256))

    # Crear los gráficos de barras para los histogramas
    fig1 = px.bar(x=np.arange(256), y=original_hist, labels={'x':'Niveles de Gris', 'y':'Frecuencia'},
                  title='Histograma Original', width=600, height=400)
    fig2 = px.bar(x=np.arange(256), y=expanded_hist, labels={'x':'Niveles de Gris', 'y':'Frecuencia'},
                  title='Histograma Expandido', width=600, height=400)

    # Mostrar los histogramas y las imágenes
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.image(expanded_image, caption='Imagen con Histograma Expandido', use_column_width=True)
    
    if st.button('Guardar Histograma'):
        st.download_button('histograma_expandido.png',fig2)

# Función para ecualizar el histograma de una imagen
def equalize_histogram(image):
    # Ecualizar el histograma
    equalized_image = ImageOps.equalize(image)

    # Obtener el histograma de la imagen original y la ecualizada
    original_hist, bins = np.histogram(np.array(image).ravel(), bins=256, range=(0,256))
    equalized_hist, bins = np.histogram(np.array(equalized_image).ravel(), bins=256, range=(0,256))

    # Crear los gráficos de barras para los histogramas
    fig1 = px.bar(x=np.arange(256), y=original_hist, labels={'x':'Niveles de Gris', 'y':'Frecuencia'},
                  title='Histograma Original', width=600, height=400)
    fig2 = px.bar(x=np.arange(256), y=equalized_hist, labels={'x':'Niveles de Gris', 'y':'Frecuencia'},
                  title='Histograma Ecualizado', width=600, height=400)

    # Mostrar los histogramas y la imagen ecualizada
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.image(equalized_image, caption='Imagen con Histograma Ecualizado', use_column_width=True)
    if st.button('Guardar Histograma'):
        st.download_button('histograma_ecualizado.png',fig2)

# Función principal
def main():
    st.title('Procesamiento digital de Imágenes')
    st.subheader('Creado por Alvaro Cáceres')
    
    with st.container():
        # Cargar imagen
        uploaded_file = st.file_uploader("Cargar Imagen", type=["jpg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Mostrar la imagen original
            st.image(image, caption='Imagen Original', use_column_width=True)
            
            # Preprocesar la imagen
            gray_image = preprocess_image(image)
            st.image(gray_image, caption='Imagen en Escala de Grises', use_column_width=True)

            # Botón para mostrar el histograma
            if st.button('Mostrar Histograma'):
                plot_histogram(gray_image)

            # Botón para expandir el histograma
            if st.button('Expandir Histograma'):
                expand_histogram(gray_image)

            # Botón para ecualizar el histograma
            if st.button('Ecualizar Histograma'):
                equalize_histogram(gray_image)

            if st.button('Guardar imagen'):
                st.download_button('imagen_convertida.png',gray_image)

if __name__ == "__main__":
    main()