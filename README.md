## ☁️ Despliegue en Streamlit Cloud

Para que la app funcione en Streamlit Cloud con Tesseract y Poppler:

1. Añade estos archivos en la raíz del repo:
   - `packages.txt`
     ```
     tesseract-ocr
     tesseract-ocr-spa
     poppler-utils
     ```
   - `requirements.txt` (pip)
     ```
     streamlit
     pytesseract
     pdf2image
     Pillow
     ```
   - `runtime.txt`
     ```
     python-3.11
     ```
   - (Opcional) `.streamlit/config.toml` para tema.

2. **Despliega** en Streamlit Cloud (New app → selecciona repo y `app.py`). La plataforma instalará automáticamente los paquetes de sistema y Python.

3. Si ves errores de idioma en Tesseract, confirma que `tesseract-ocr-spa` está presente. Si persiste, fija `TESSDATA_PREFIX` en el código (comentado en `app.py`).

4. Ajusta en la barra lateral:
   - **PSM**: 6 (bloques), 3 (página completa), 11/12 (texto disperso).
   - **Idiomas**: `spa+eng` si hay mezcla.
   - Preprocesado: grises + autocontraste + nitidez; usa binarización si hay mucho ruido.
