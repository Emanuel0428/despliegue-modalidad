# -*- coding: utf-8 -*-
"""Despliegue_Modalidad_ValidacionCruzada

# Despliegue

- Cargamos el modelo
- Capturamos los datos futuros (interfaz Streamlit)
- Preparar los datos futuros: dummies (no se normaliza, XGBoost usa árboles)
- Aplicamos el modelo para la predicción
"""

#Cargamos librerías principales
import html
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

#Cargamos el modelo
import pickle
filename = 'modelo-cla.pkl'
modelo, labelencoder, variables = pickle.load(open(filename, 'rb'))

#Variables categóricas que usa el modelo (las demás se eliminaron en la selección de factores)
categoricas = ['Comuna', 'Ciudad Nacimiento', 'Nombre Programa', 'Facultad', 'Area Conocimiento']

#Algunas dummies traen espacios, saltos de línea o tildes codificadas distinto ('Comuna_El Peñol\n'):
#los valores se comparan normalizados (sin tildes, minúsculas, espacios simples) y al final se devuelven los nombres originales
def normalizar(texto):
    texto = unicodedata.normalize('NFKD', str(texto))
    texto = ''.join(ch for ch in texto if not unicodedata.combining(ch))
    return ' '.join(texto.lower().split())

def partir(columna):
    var = next(v for v in categoricas if columna.startswith(v + '_'))
    return var, columna[len(var) + 1:]

claves = [f'{var}_{normalizar(valor)}' for var, valor in map(partir, variables)]

#Opciones de cada variable = categorías que el modelo conoce (se extraen de las dummies)
#"Otra" deja todas las dummies en 0 (categoría base)
def opciones(var):
    return [valor.strip() for v, valor in map(partir, variables) if v == var] + ['Otra']

#Preparación de datos futuros (sirve para una ficha o para un Excel con muchas filas)
def preparar(data):
    data_preparada = data[categoricas].apply(lambda s: s.map(normalizar))
    #En despliegue drop_first= False
    data_preparada = pd.get_dummies(data_preparada, columns=categoricas, drop_first=False, dtype=int)
    #Se adicionan las columnas faltantes (y se descartan categorías que el modelo no conoce)
    data_preparada = data_preparada.reindex(columns=claves, fill_value=0)
    data_preparada.columns = variables
    #No se normaliza: las predictoras son dummies 0/1 y el modelo es XGBoost
    return data_preparada

#Etiquetas legibles y color de tinta del sello para cada modalidad
etiquetas = {'Nombre Programa': 'Programa', 'Facultad': 'Facultad', 'Area Conocimiento': 'Área de conocimiento',
             'Comuna': 'Comuna', 'Ciudad Nacimiento': 'Ciudad de nacimiento'}
tintas = {'Presencial': '#2346D8', 'Semipresencial': '#8A3FD1', 'Virtual': '#0B8F6B'}


#Se crea interfaz gráfica con streamlit para captura de los datos
st.set_page_config(page_title="Sello de modalidad", page_icon="🎓", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Instrument+Sans:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap');
:root { --papel:#E3E7F3; --ficha:#FFFFFF; --tinta:#1A1D3A; --gris:#5B6285; --linea:#D2D8EC; --marcador:#FFE14D; }
.stApp { background: var(--papel) repeating-linear-gradient(0deg, transparent 0 31px, var(--linea) 31px 32px); color: var(--tinta); font-family: 'Instrument Sans', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3rem; max-width: 760px; }

.eyebrow { font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--gris); }
.titulo { font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:clamp(2.1rem, 7vw, 3.6rem); line-height:1; letter-spacing:-.035em; margin:.6rem 0 .9rem; color:var(--tinta); overflow-wrap:anywhere; }
.bajada { font-size:1.05rem; color:var(--gris); max-width:34rem; margin-bottom:1.6rem; }

.st-key-ficha { background:var(--ficha); border:1.5px solid var(--tinta) !important; border-radius:6px; padding:1.4rem 1.4rem .6rem; box-shadow:6px 6px 0 var(--tinta); }
.grupo { font-family:'JetBrains Mono', monospace; font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:var(--tinta); border-bottom:2px dashed var(--linea); padding-bottom:.35rem; margin:.4rem 0 .6rem; }
[data-testid="stWidgetLabel"] p { font-family:'Instrument Sans', sans-serif; font-weight:600; color:var(--tinta) !important; }
[data-baseweb="select"] > div { background:#F6F7FC !important; border:1.5px solid var(--tinta) !important; border-radius:4px !important; }

.st-key-sellar button, .st-key-descargar button { width:100%; background:var(--marcador) !important; color:var(--tinta) !important; border:1.5px solid var(--tinta); border-radius:4px; padding:.75rem 1rem; font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:1.1rem; box-shadow:4px 4px 0 var(--tinta); transition:transform .12s ease, box-shadow .12s ease; }
.st-key-sellar button p, .st-key-descargar button p { font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:1.1rem; }
.st-key-sellar button:hover, .st-key-descargar button:hover { color:var(--tinta); border-color:var(--tinta); transform:translate(-1px,-1px); box-shadow:5px 5px 0 var(--tinta); }
.st-key-sellar button:active, .st-key-descargar button:active { transform:translate(3px,3px); box-shadow:1px 1px 0 var(--tinta); }
.st-key-sellar button:focus-visible, .st-key-descargar button:focus-visible { outline:3px solid var(--tinta); outline-offset:3px; }
.nota { font-family:'JetBrains Mono', monospace; font-size:.72rem; color:var(--gris); margin:.6rem 0 1.8rem; }

.resultado { background:var(--ficha); border:1.5px solid var(--tinta); border-radius:6px; box-shadow:6px 6px 0 var(--tinta); overflow:hidden; }
.resultado-cab { display:flex; justify-content:space-between; padding:.7rem 1.4rem; border-bottom:2px dashed var(--linea); }
.resultado-cuerpo { display:flex; gap:1.5rem; align-items:center; padding:1.4rem; flex-wrap:wrap; }
.perfil { flex:1 1 240px; margin:0; }
.perfil dt { font-family:'JetBrains Mono', monospace; font-size:.66rem; letter-spacing:.12em; text-transform:uppercase; color:var(--gris); }
.perfil dd { margin:0 0 .55rem; font-weight:600; color:var(--tinta); }

.sello { flex:0 0 auto; margin:0 auto; color:var(--ink); border:5px double var(--ink); border-radius:10px; padding:.8rem 1.3rem; text-align:center; transform:rotate(-9deg); mix-blend-mode:multiply;
  -webkit-mask-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2'/><feColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 -.55'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  mask-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2'/><feColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 -.55'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  animation:golpe .45s cubic-bezier(.2,.9,.3,1.2) both; }
.sello small { display:block; font-family:'JetBrains Mono', monospace; font-size:.68rem; letter-spacing:.2em; font-weight:700; }
.sello b { display:block; font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:clamp(1.7rem, 5vw, 2.4rem); letter-spacing:.02em; text-transform:uppercase; line-height:1.05; margin:.15rem 0; }
.sello-vacio { flex:0 0 auto; margin:0 auto; border:2px dashed var(--linea); border-radius:10px; padding:1.4rem 1.2rem; max-width:220px; text-align:center; color:var(--gris); font-size:.9rem; }
@keyframes golpe { 0% { opacity:0; transform:rotate(-9deg) scale(1.9); } 70% { opacity:1; transform:rotate(-9deg) scale(.96); } 100% { transform:rotate(-9deg) scale(1); } }

.probs { padding:0 1.4rem 1.3rem; }
.fila { display:grid; grid-template-columns:8.5rem 1fr 3.2rem; gap:.7rem; align-items:center; margin-top:.45rem; font-size:.9rem; }
.barra { height:10px; background:#EEF0F8; border-radius:2px; overflow:hidden; }
.barra i { display:block; height:100%; background:var(--ink); }
.pct { font-family:'JetBrains Mono', monospace; font-size:.8rem; text-align:right; }
.gana { font-weight:700; }

[data-baseweb="tab"] p { font-family:'JetBrains Mono', monospace; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:var(--tinta); }
[data-baseweb="tab-highlight"] { background:var(--tinta); }
.conteo { display:flex; gap:1rem; flex-wrap:wrap; margin:.4rem 0 1.2rem; }
.mini-sello { color:var(--ink); border:3px double var(--ink); border-radius:8px; padding:.4rem .9rem; text-align:center; transform:rotate(-4deg); background:var(--ficha); }
.mini-sello b { display:block; font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:1.6rem; line-height:1; }
.mini-sello small { font-family:'JetBrains Mono', monospace; font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; }

@media (max-width: 560px) { .fila { grid-template-columns:6.5rem 1fr 3rem; } .st-key-ficha { padding:1rem 1rem .4rem; } }
@media (prefers-reduced-motion: reduce) { .sello { animation:none; } .st-key-sellar button { transition:none; } }
</style>
""", unsafe_allow_html=True)

#Encabezado: las tres modalidades posibles, cada una con su color de sello
st.markdown(
    '<div class="eyebrow">Matrícula 2026-1 · modelo XGBoost</div>'
    '<div class="titulo" role="heading" aria-level="1">¿<span style="color:#2346D8">Presencial</span>, <span style="color:#8A3FD1">semipresencial</span> '
    'o <span style="color:#0B8F6B">virtual</span>?</div>'
    '<div class="bajada">Llena la ficha con los datos del programa y del aspirante. El modelo sella la modalidad más probable.</div>',
    unsafe_allow_html=True)

una, varias = st.tabs(['Una ficha', 'Varias fichas (Excel)'])

with una:
    entradas = {}
    with st.container(border=True, key="ficha"):
        st.markdown('<div class="grupo">Del programa</div>', unsafe_allow_html=True)
        entradas['Nombre Programa'] = st.selectbox(etiquetas['Nombre Programa'], opciones('Nombre Programa'))
        col1, col2 = st.columns(2)
        entradas['Facultad'] = col1.selectbox(etiquetas['Facultad'], opciones('Facultad'))
        entradas['Area Conocimiento'] = col2.selectbox(etiquetas['Area Conocimiento'], opciones('Area Conocimiento'))
        st.markdown('<div class="grupo">Del aspirante</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        entradas['Comuna'] = col3.selectbox(etiquetas['Comuna'], opciones('Comuna'))
        entradas['Ciudad Nacimiento'] = col4.selectbox(etiquetas['Ciudad Nacimiento'], opciones('Ciudad Nacimiento'))

    #Dataframe
    data = pd.DataFrame([entradas], columns=categoricas)  #Dataframe con los mismos nombres de variables

    #Se realiza la preparación de datos
    data_preparada = preparar(data)

    # Predicción
    sellar = st.button('Sellar predicción', key="sellar")

    # Recordar medida de error del modelo
    st.markdown('<div class="nota">f1_macro 0.78 en validación cruzada de 10 particiones. Úsalo como orientación, no como decisión final.</div>',
                unsafe_allow_html=True)

    #Perfil ingresado, en el orden de la ficha
    perfil = ''.join(f'<dt>{etiquetas[v]}</dt><dd>{html.escape(entradas[v])}</dd>'
                     for v in ['Nombre Programa', 'Facultad', 'Area Conocimiento', 'Comuna', 'Ciudad Nacimiento'])

    if sellar:
        #Hacemos la predicción con XGBoost
        Y_pred = modelo.predict(data_preparada)

        #Se convierte 0/1/2 a la etiqueta original (Presencial/Semipresencial/Virtual)
        Y_pred_etiqueta = labelencoder.inverse_transform(Y_pred)
        data['Prediccion'] = Y_pred_etiqueta
        modalidad = Y_pred_etiqueta[0]

        #Probabilidad de cada modalidad
        probabilidades = modelo.predict_proba(data_preparada)[0]
        filas = ''.join(
            f'<div class="fila{" gana" if clase == modalidad else ""}" style="--ink:{tintas[clase]}">'
            f'<span>{clase}</span><div class="barra"><i style="width:{p * 100:.1f}%"></i></div>'
            f'<span class="pct">{p * 100:.0f}%</span></div>'
            for clase, p in zip(labelencoder.classes_, probabilidades))

        sello = (f'<div class="sello" style="--ink:{tintas[modalidad]}"><small>MODALIDAD</small>'
                 f'<b>{modalidad}</b><small>{probabilidades.max() * 100:.0f}% SEGURO</small></div>')
        cabecera = 'Sellado'
        pie = f'<div class="probs"><div class="eyebrow">Probabilidad por modalidad</div>{filas}</div>'
    else:
        sello = '<div class="sello-vacio">Aún sin sello. Completa la ficha y presiona <b>Sellar predicción</b>.</div>'
        cabecera = 'Pendiente'
        pie = ''

    #Resultado: la ficha con el sello de la modalidad
    st.markdown(
        f'<div class="resultado"><div class="resultado-cab"><span class="eyebrow">Ficha del aspirante</span>'
        f'<span class="eyebrow">{cabecera}</span></div>'
        f'<div class="resultado-cuerpo"><dl class="perfil">{perfil}</dl>{sello}</div>{pie}</div>',
        unsafe_allow_html=True)

with varias:
    #Cargamos los datos futuros desde un Excel
    st.markdown('<div class="bajada">Sube un Excel con una fila por aspirante y las columnas '
                '<b>Comuna</b>, <b>Ciudad Nacimiento</b>, <b>Nombre Programa</b>, <b>Facultad</b> y <b>Area Conocimiento</b>. '
                'Se sella la modalidad de todas las filas.</div>', unsafe_allow_html=True)
    archivo = st.file_uploader('Excel de datos futuros', type=['xlsx'])

    if archivo is not None:
        data_futura = pd.read_excel(archivo)
        faltantes = [c for c in categoricas if c not in data_futura.columns]

        if faltantes:
            st.error(f'Al archivo le faltan las columnas: {", ".join(faltantes)}. Agrégalas con ese nombre exacto y vuelve a subirlo.')
        else:
            #Se realiza la preparación de datos y la predicción de todas las filas
            data_preparada_futura = preparar(data_futura)
            Y_pred = modelo.predict(data_preparada_futura)
            data_futura['Prediccion'] = labelencoder.inverse_transform(Y_pred)
            data_futura['Seguridad'] = modelo.predict_proba(data_preparada_futura).max(axis=1).round(2)

            #Resumen: cuántas filas quedaron en cada modalidad
            conteo = data_futura['Prediccion'].value_counts()
            st.markdown('<div class="conteo">' + ''.join(
                f'<div class="mini-sello" style="--ink:{tintas[c]}"><b>{conteo.get(c, 0)}</b><small>{c}</small></div>'
                for c in labelencoder.classes_) + '</div>', unsafe_allow_html=True)

            st.dataframe(data_futura, use_container_width=True)

            #Descarga del Excel con las predicciones
            salida = io.BytesIO()
            data_futura.to_excel(salida, index=False)
            st.download_button('Descargar Excel con predicciones', salida.getvalue(),
                               file_name='predicciones_modalidad.xlsx', key="descargar")

    st.markdown('<div class="nota">f1_macro 0.78 en validación cruzada de 10 particiones. Úsalo como orientación, no como decisión final.</div>',
                unsafe_allow_html=True)
