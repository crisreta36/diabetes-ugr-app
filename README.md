# Clasificación de diabetes — TP UGR

Demostración académica en Streamlit basada en el modelo Random Forest exportado del notebook grupal.

## Publicación

Subir `streamlit_app.py`, `requirements.txt` y `modelo_diabetes.joblib` a la raíz del repositorio público. En Streamlit Community Cloud seleccionar el repositorio, la rama `main` y el archivo principal `streamlit_app.py`.

El modelo se carga desde un archivo `.joblib` generado por el propio grupo. **No cargar archivos `.joblib` de fuentes desconocidas**. Confirmar que el preprocesamiento del notebook coincide con la entrada de esta aplicación y comprobar la predicción con un caso de referencia.

La puntuación del modelo no es una probabilidad clínica calibrada y la aplicación no reemplaza la evaluación médica.
