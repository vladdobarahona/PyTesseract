
# app.py
# Streamlit + Tesseract OCR para extraer texto de PDFs e imágenes
# Autor: Vladimir Alonso B. P. (para uso personal)
# Ejecuta: streamlit run app.py

import io
import os
from typing import List, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract

# PDF -> imágenes
# Nota: pdf2image requiere Poppler instalado en el sistema.
from pdf2image import convert_from_bytes

# -----------------------------
# Configuración de la interfaz
# -----------------------------
st.set_page_config(page_title="OCR con Tesseract (PDF/Imagen)", page_icon="📄", layout="wide")
st.title("📄 OCR con Tesseract (PDF/Imagen) – Streamlit")
st.caption("Sube un PDF o una imagen y obtén el texto extraído. Incluye preprocesamiento y corrección de rotación.")

# -----------------------------
# Sidebar: opciones
# -----------------------------
st.sidebar.header("⚙️ Opciones de OCR y preprocesamiento")

# Ruta manual de Tesseract (útil en Windows)
tesseract_path = st.sidebar.text_input(
    "Ruta de Tesseract (opcional, solo si no se detecta automáticamente)",
    value="",
    help="Ejemplo Windows: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)
if tesseract_path.strip():
    pytesseract.pytesseract.tesseract_cmd = tesseract_path.strip()

# Idiomas
langs = st.sidebar.multiselect(
    "Idiomas (tesseract 'lang' instalados)",
    options=["spa", "eng", "por", "fra", "deu"],
    default=["spa", "eng"],
    help="Selecciona los idiomas que tengas instalados en Tesseract (tessdata)."
)
lang_code = "+".join(langs) if langs else "spa"

# OEM y PSM
oem = st.sidebar.selectbox("OEM (motor OCR)", options=[0, 1, 2, 3], index=3,
                           help="0: Original, 1: Neural LSTM, 2: Combinado, 3: Default según Tesseract.")
psm = st.sidebar.selectbox(
    "PSM (Page Segmentation Mode)",
    options=[3, 4, 6, 11, 12, 13],
    index=6,
    help="3: Full page; 6: Asignación de bloques; 11: Sparse text; 13: Raw line; etc."
)

# Preprocesamiento
do_grayscale = st.sidebar.checkbox("Escala de grises", value=True)
do_autocontrast = st.sidebar.checkbox("Aumentar contraste", value=True)
do_sharpen = st.sidebar.checkbox("Nitidez", value=True)
do_threshold = st.sidebar.checkbox("Binarización (Umbral)", value=True)
do_denoise = st.sidebar.checkbox("Reducción de ruido (mediana)", value=True)
do_deskew = st.sidebar.checkbox("Detectar y corregir rotación", value=True)

dpi = st.sidebar.slider("DPI (solo PDFs)", min_value=200, max_value=400, value=300, step=50)

# -----------------------------
# Utilidades
# -----------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    """Aplica preprocesamiento básico para mejorar el OCR."""
    if do_grayscale and img.mode != "L":
        img = ImageOps.grayscale(img)

    if do_autocontrast:
        img = ImageOps.autocontrast(img)

    if do_sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))

    if do_threshold:
        # Umbral simple adaptado (para modo 'L' preferible)
        if img.mode != "L":
            tmp = ImageOps.grayscale(img)
        else:
            tmp = img
        # Umbral por autocontrast y punto
        # Usamos un umbral fijo conservador; ajusta según calidad
        img = tmp.point(lambda p: 255 if p > 160 else 0)

    if do_denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    return img

def try_deskew(img: Image.Image) -> Image.Image:
    """Usa OSD (Orientation & Script Detection) para estimar rotación y corregirla."""
    try:
        osd = pytesseract.image_to_osd(img)
        # Ejemplo de salida: "Rotate: 90\nOrientation: ...\n"
        angle = 0
        for line in osd.splitlines():
            if line.lower().startswith("rotate"):
                angle = int(line.split(":")[1].strip())
                break
        if angle != 0:
            img = img.rotate(360 - angle, expand=True, fillcolor=255)
    except Exception:
        pass
    return img

def ocr_image(img: Image.Image, lang: str, oem_: int, psm_: int) -> str:
    """Ejecuta OCR sobre una imagen con configuración personalizada."""
    config = f"--oem {oem_} --psm {psm_}"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text

def pdf_to_images(pdf_bytes: bytes, dpi_: int) -> List[Image.Image]:
    """Convierte PDF (bytes) a imágenes PIL. Requiere Poppler instalado."""
    pages = convert_from_bytes(pdf_bytes, dpi=dpi_, fmt="png")
    return pages

def make_txt_download(name: str, content: str):
    st.download_button(
        label=f"💾 Descargar texto ({name}.txt)",
        data=content.encode("utf-8"),
        file_name=f"{name}.txt",
        mime="text/plain"
    )

# -----------------------------
# Carga de archivo
# -----------------------------
uploaded = st.file_uploader(
    "Sube un PDF o una imagen (PNG/JPG/TIFF)",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=False
)

if uploaded is None:
    st.info("👉 Sube un archivo para comenzar.")
else:
    file_name = uploaded.name
    st.subheader(f"Archivo: {file_name}")

    # Detecta el tipo y genera lista de imágenes
    images: List[Image.Image] = []
    try:
        if file_name.lower().endswith(".pdf"):
            # PDF -> páginas (imágenes)
            pdf_bytes = uploaded.read()
            with st.spinner("Convirtiendo PDF a imágenes..."):
                images = pdf_to_images(pdf_bytes, dpi_=dpi)
        else:
            # Imagen única
            images = [Image.open(uploaded)]
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        st.stop()

    st.write(f"📄 Páginas/Imágenes detectadas: {len(images)}")

    all_text = []
    for idx, img in enumerate(images, start=1):
        st.markdown(f"### Página/Imagen {idx}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"Original {idx}", use_column_width=True)

        # Preprocesamiento y deskew
        proc = preprocess_image(img)
        if do_deskew:
            proc = try_deskew(proc)

        with col2:
            st.image(proc, caption=f"Preprocesada {idx}", use_column_width=True)

        # OCR
        with st.spinner("Ejecutando OCR..."):
            text = ocr_image(proc, lang=lang_code, oem_=oem, psm_=psm)

        # Mostrar resultado
        with st.expander("📝 Texto extraído"):
            st.code(text)

        all_text.append(text)

    # Texto consolidado
    merged_text = "\n\n".join(all_text).strip()
    st.markdown("## Resultado consolidado")
    st.text_area("Texto", value=merged_text, height=250)
    make_txt_download(os.path.splitext(file_name)[0], merged_text)

    # (Opcional) Generar PDF con texto incrustado (searchable)
    # Nota: Requiere Tesseract 4+ y puede ser pesado en documentos largos.
    if st.checkbox("Generar PDF 'searchable' (por página)"):
        for i, img in enumerate(images, start=1):
            try:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf')
                st.download_button(
                    label=f"💾 Descargar página {i} OCR-PDF",
                    data=pdf_bytes,
                    file_name=f"{os.path.splitext(file_name)[0]}_ocr_p{i}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.warning(f"No se pudo generar PDF OCR para la página {i}: {e}")
