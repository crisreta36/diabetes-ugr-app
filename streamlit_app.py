"""Demostración académica: clasificación de diabetes (UGR)."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

COLUMNAS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
CEROS_FALTANTES = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
UMBRAL = 0.40

st.set_page_config(page_title='Clasificación de diabetes | TP UGR', page_icon='📊', layout='centered')

@st.cache_resource
def cargar_modelo():
    return joblib.load(Path(__file__).with_name('modelo_diabetes.joblib'))

st.title('Clasificación de diabetes · TP Ingeniería de Software')
st.write('**Demostración académica de CRISP-DM** con el modelo Random Forest del trabajo grupal.')
st.warning('No es un diagnóstico médico ni una herramienta clínica validada. No ingreses datos personales identificatorios.')

with st.form('clasificacion'):
    col1, col2 = st.columns(2)
    with col1:
        embarazos = st.number_input('Embarazos (cantidad)', min_value=0, value=1, step=1)
        glucosa = st.number_input('Glucosa (mg/dL)', min_value=0.0, value=120.0)
        presion = st.number_input('Presión diastólica (mm Hg)', min_value=0.0, value=70.0)
        pliegue = st.number_input('Pliegue tricipital (mm)', min_value=0.0, value=0.0)
    with col2:
        insulina = st.number_input('Insulina (µU/mL)', min_value=0.0, value=0.0)
        imc = st.number_input('IMC (kg/m²)', min_value=0.0, value=28.0)
        antecedentes = st.number_input('Índice de antecedentes familiares', min_value=0.0, value=0.4, step=0.01, format='%.2f')
        edad = st.number_input('Edad (años)', min_value=1, max_value=120, value=35, step=1)
    enviar = st.form_submit_button('Ejecutar clasificación', type='primary')

st.caption('En glucosa, presión, pliegue, insulina e IMC, un valor 0 se interpreta como dato no medido, según la preparación del notebook.')

if enviar:
    valores = [embarazos, glucosa, presion, pliegue, insulina, imc, antecedentes, edad]
    if not all(np.isfinite(float(v)) for v in valores):
        st.error('Completá los ocho campos con valores numéricos válidos.')
    else:
        fila = pd.DataFrame([valores], columns=COLUMNAS)
        fila[CEROS_FALTANTES] = fila[CEROS_FALTANTES].replace(0, np.nan)
        try:
            modelo = cargar_modelo()
            puntuacion = float(modelo.predict_proba(fila)[0, 1])
            if puntuacion >= UMBRAL:
                st.error('Clasificación positiva del modelo')
            else:
                st.success('Clasificación negativa del modelo')
            st.metric('Puntuación del modelo (no calibrada)', f'{puntuacion:.3f}')
            st.caption(f'Umbral utilizado: {UMBRAL:.2f}. El resultado experimental no permite confirmar ni descartar diabetes.')
        except Exception as e:
            st.error('No se pudo cargar el modelo o generar la clasificación.')
            st.exception(e)

st.divider()
st.caption('Fuente: Pima Indians Diabetes Database. Proyecto académico sin validación clínica.')
