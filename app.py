# app.py
# Streamlit + Tesseract OCR para extraer texto de PDFs e imágenes
# Autor: Vladimir Alonso B. P. (para uso personal)
# Ejecuta: streamlit run app.py

import os
import io
import shutil
from typing import List

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# PDF -> imágenes (requiere Poppler instalado en el sistema)
from pdf2image import convert_from_bytes


# -----------------------------
# Configuración de la interfaz
# -----------------------------
st.set_page_config(page_title="OCR con Tesseract (PDF/Imagen)", page_icon="📄", layout="wide")
st.title("📄 OCR con Tesseract (PDF/Imagen) – Streamlit")
st.caption("Sube un PDF o una imagen y obtén el texto extraído. Incluye preprocesamiento y corrección de rotación.")


# -----------------------------
# Verificación de dependencias de sistema
# -----------------------------
def _maybe_set_tessdata_prefix():
    """Intenta establecer TESSDATA_PREFIX si no está configurado y existe un camino conocido."""
    if "TESSDATA_PREFIX" in os.environ:
        return os.environ["TESSDATA_PREFIX"]
    # Rutas comunes en Debian/Ubuntu (usadas por los contenedores de Streamlit Cloud)
    candidates = [
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/tessdata",
    ]
    for path in candidates:
        if os.path.isdir(path):
            os.environ["TESSDATA_PREFIX"] = path
            return path
    return None


def ensure_tesseract() -> bool:
    """Verifica que Tesseract esté disponible y configura pytesseract."""
    tpath = shutil.which("tesseract")
    if not tpath:
        st.error(
            "❌ Tesseract no está instalado o no está en PATH.\n\n"
            "En Streamlit Cloud añade 'packages.txt' con:\n"
            "  - tesseract-ocr\n  - tesseract-ocr-spa\n  - poppler-utils"
        )
        return False

    pytesseract.pytesseract.tesseract_cmd = tpath
    # Intenta fijar TESSDATA_PREFIX si no está disponible
    _maybe_set_tessdata_prefix()

    # Mostrar estado al usuario
    st.caption(f"✅ Tesseract detectado en: {tpath}")
    return True


def ensure_poppler() -> bool:
    """Verifica que Poppler esté disponible (pdftoppm) para pdf2image."""
    if shutil.which("pdftoppm"):
        return True
    st.warning("⚠️ Poppler no está disponible. Añade 'poppler-utils' en packages.txt.")
    return False


# Verificar Tesseract desde el inicio
if not ensure_tesseract():
    st.stop()


# -----------------------------
# Sidebar: opciones de OCR y preprocesamiento
# -----------------------------
st.sidebar.header("⚙️ Opciones de OCR y preprocesamiento")

# Ruta manual de Tesseract (útil en Windows local; en Cloud normalmente NO es necesaria)
tesseract_path = st.sidebar.text_input(
    "Ruta de Tesseract (opcional, solo si no se detecta automáticamente)",
    value="",
    help="Ejemplo Windows: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)
if tesseract_path.strip():
    pytesseract.pytesseract.tesseract_cmd = tesseract_path.strip()
    st.caption(f"🔧 Usando ruta de Tesseract proporcionada: {tesseract_path.strip()}")

# Idiomas disponibles (mostrados si Tesseract puede listarlos)
available_langs = []
try:
    available_langs = pytesseract.get_languages(config="")
except Exception:
    # Si falla, no bloquear; probablemente por TESSDATA_PREFIX
    pass

lang_options = ["spa", "eng", "por", "fra", "deu"]
langs = st.sidebar.multiselect(
    "Idiomas (tesseract 'lang' instalados)",
    options=lang_options,
    default=["spa", "eng"],
    help="Selecciona los idiomas instalados en Tesseract (tessdata)."
)
lang_code = "+".join(langs) if langs else "spa"

# Aviso si algún idioma no está en la instalación actual
if available_langs:
    missing = [l for l in langs if l not in available_langs]
    if missing:
        st.warning(
            f"Los siguientes idiomas seleccionados no aparecen en la instalación: {missing}. "
            f"Idiomas detectados: {available_langs}. "
            f"En Streamlit Cloud, incluye 'tesseract-ocr-spa' para español en packages.txt."
        )

# OEM y PSM
oem = st.sidebar.selectbox(
    "OEM (motor OCR)",
    options=[0, 1, 2, 3],
    index=3,
    help="0: Original; 1: LSTM; 2: Combinado; 3: Default."
)

psm_options = [3, 4, 6, 11, 12, 13]
default_psm_value = 6
psm_default_index = psm_options.index(default_psm_value) if default_psm_value in psm_options else 0
psm = st.sidebar.selectbox(
    "PSM (Page Segmentation Mode)",
    options=psm_options,
    index=psm_default_index,
    help="3: Página completa; 6: Bloques; 11: Texto disperso; 12: Disperso + OSD; 13: Línea cruda."
)

# Preprocesamiento
do_grayscale = st.sidebar.checkbox("Escala de grises", value=True)
do_autocontrast = st.sidebar.checkbox("Aumentar contraste", value=True)
do_sharpen = st.sidebar.checkbox("Nitidez", value=True)
do_threshold = st.sidebar.checkbox("Binarización (Umbral)", value=True)
do_denoise = st.sidebar.checkbox("Reducción de ruido (mediana)", value=True)
do_deskew = st.sidebar.checkbox("Detectar y corregir rotación (OSD)", value=True)

dpi = st.sidebar.slider("DPI (solo PDFs)", min_value=200, max_value=400, value=300, step=50)


# -----------------------------
# Utilidades
# -----------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    """Aplica preprocesamiento básico para mejorar el OCR."""
    try:
        # Escala de grises
        if do_grayscale and img.mode != "L":
            img = ImageOps.grayscale(img)

        # Contraste automático
        if do_autocontrast:
            img = ImageOps.autocontrast(img)

        # Nitidez
        if do_sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))

        # Binarización (umbral sencillo)
        if do_threshold:
            # Asegurar modo L para umbral
            tmp = img if img.mode == "L" else ImageOps.grayscale(img)
            # Umbral fijo (ajustable según calidad de escaneo)
            img = tmp.point(lambda p: 255 if p > 160 else 0)

        # Reducción de ruido
        if do_denoise:
            img = img.filter(ImageFilter.MedianFilter(size=3))

        return img
    except Exception as e:
        st.warning(f"Preprocesamiento: {e}")
        return img


def try_deskew(img: Image.Image) -> Image.Image:
    """Usa OSD (Orientation & Script Detection) para estimar rotación y corregirla."""
    if not do_deskew:
        return img
    try:
        osd = pytesseract.image_to_osd(img)
        angle = 0
        for line in osd.splitlines():
            if line.lower().startswith("rotate"):
                angle = int(line.split(":")[1].strip())
                break
        if angle != 0:
            # Rotar en sentido contrario para corregir
            fillcolor = 255 if img.mode in ("L", "1") else (255, 255, 255)
            img = img.rotate(360 - angle, expand=True, fillcolor=fillcolor)
    except Exception as e:
        # OSD puede fallar si la imagen es muy pequeña o si hay poco texto.
        st.info(f"No se pudo determinar rotación automáticamente (OSD): {e}")
    return img


def ocr_image(img: Image.Image, lang: str, oem_: int, psm_: int) -> str:
    """Ejecuta OCR sobre una imagen con configuración personalizada."""
    config = f"--oem {oem_} --psm {psm_}"
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        return text
    except pytesseract.TesseractError as te:
        st.error(f"Error de Tesseract: {te}")
        return ""
    except Exception as e:
        st.error(f"OCR: {e}")
        return ""


def pdf_to_images(pdf_bytes: bytes, dpi_: int) -> List[Image.Image]:
    """Convierte PDF (bytes) a imágenes PIL. Requiere Poppler instalado."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi_, fmt="png")
        return pages
    except Exception as e:
        st.error(f"No se pudo convertir el PDF a imágenes. Verifica Poppler: {e}")
        return []


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
            # Verificar Poppler antes de convertir
            if not ensure_poppler():
                st.stop()
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

    if not images:
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
    st.markdown("---")
    st.markdown("### 📑 Generar PDF con texto incrustado por página (opcional)")
    if st.checkbox("Crear PDF 'searchable' por página"):
        for i, img in enumerate(images, start=1):
            try:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf')
                st.download_button(
                    label=f"💾 Descargar página {i} OCR-PDF",
                    data=pdf_bytes,
                    file_name=f"{os.path.splitext(file_name)[0]}_ocr_p{i}.pdf",
                    mime="application/pdf",
                    key=f"pdf_dl_{i}"
                )
            except Exception as e:
                st.warning(f"No se pudo generar PDF OCR para la página {i}: {e}")
