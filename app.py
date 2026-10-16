"""Reconocimiento de digitos escritos a mano con una red convolucional (MNIST).

Ejecutar en local:
    streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import preprocesamiento as pre

RAIZ = Path(__file__).parent
CARPETA_MODELO = RAIZ / "modelo"
CARPETA_EJEMPLOS = RAIZ / "ejemplos"
CONFIANZA_BAJA = 0.60
MAX_DIGITOS_DETALLE = 8

st.set_page_config(page_title="Dígitos con CNN", page_icon="🔢", layout="wide")

# Colores de las graficas (paleta validada para daltonismo, pasos claro/oscuro)
TEMA_OSCURO = getattr(st.context.theme, "type", None) == "dark"
AZUL = "#3987e5" if TEMA_OSCURO else "#2a78d6"
NARANJA = "#d95926" if TEMA_OSCURO else "#eb6834"
GRIS = "#52514e" if TEMA_OSCURO else "#c3c2b7"
RAMPA_AZUL = ["#184f95", "#86b6ef"] if TEMA_OSCURO else ["#cde2fb", "#0d366b"]
# Texto sobre las celdas de los mapas de calor (valor alto / valor bajo)
TEXTO_ALTO, TEXTO_BAJO = ("black", "white") if TEMA_OSCURO else ("white", "black")
NEUTRO = "#383835" if TEMA_OSCURO else "#f0efec"  # celdas sin datos (cero errores)
SUPERFICIE = "#0e1117" if TEMA_OSCURO else "#ffffff"
TEXTO = "#fafafa" if TEMA_OSCURO else "#31333f"  # color de texto de Streamlit


# --------------------------------------------------------------------------
# Carga del modelo y de los resultados del entrenamiento
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando la red neuronal...")
def cargar_modelo():
    import tensorflow as tf

    modelo = tf.keras.models.load_model(CARPETA_MODELO / "numeros_cnn.keras")
    # Submodelo que devuelve la salida de las dos capas convolucionales
    visor = tf.keras.Model(
        inputs=modelo.inputs[0],
        outputs=[modelo.get_layer("conv1").output, modelo.get_layer("conv2").output],
    )
    return modelo, visor


@st.cache_data
def leer_json(nombre):
    ruta = CARPETA_MODELO / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


@st.cache_data
def leer_errores():
    ruta = CARPETA_MODELO / "errores_prueba.npz"
    return dict(np.load(ruta)) if ruta.exists() else None


def predecir(lote):
    modelo, _ = cargar_modelo()
    return modelo(np.asarray(lote, np.float32), training=False).numpy()


# --------------------------------------------------------------------------
# Graficas e imagenes
# --------------------------------------------------------------------------
def grafica_probabilidades(prob):
    datos = pd.DataFrame({
        "Dígito": [str(i) for i in range(10)],
        "Probabilidad": prob,
        "Elegido": prob == prob.max(),
    })
    base = alt.Chart(datos).encode(
        x=alt.X("Dígito:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        # Dominio hasta 1.15 para que la etiqueta del 100 % quepa sobre la barra
        y=alt.Y("Probabilidad:Q", scale=alt.Scale(domain=[0, 1.15]),
                axis=alt.Axis(format="%", values=[0, 0.5, 1], title=None)),
        tooltip=[alt.Tooltip("Dígito:N"), alt.Tooltip("Probabilidad:Q", format=".2%")],
    )
    barras = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.condition("datum.Elegido", alt.value(AZUL), alt.value(GRIS)),
    )
    etiqueta = base.transform_filter("datum.Elegido").mark_text(dy=-8, fontWeight="bold", color=TEXTO).encode(
        text=alt.Text("Probabilidad:Q", format=".0%"),
    )
    return (barras + etiqueta).properties(height=170)


def ampliar(img28, lado=112):
    """28x28 en [0, 1] -> imagen grande con pixeles visibles."""
    img = (np.clip(img28.squeeze(), 0, 1) * 255).astype(np.uint8)
    return cv2.resize(img, (lado, lado), interpolation=cv2.INTER_NEAREST)


def mosaico(mapas, columnas):
    """(alto, ancho, canales) -> una sola imagen con todos los mapas en rejilla."""
    h, w, c = mapas.shape
    filas = int(np.ceil(c / columnas))
    rejilla = np.zeros((filas * (h + 1), columnas * (w + 1)), np.float32)
    for i in range(c):
        m = mapas[..., i]
        rango = m.max() - m.min()
        m = (m - m.min()) / rango if rango > 0 else np.zeros_like(m)
        f, k = divmod(i, columnas)
        rejilla[f * (h + 1):f * (h + 1) + h, k * (w + 1):k * (w + 1) + w] = m
    escala = max(1, 360 // rejilla.shape[1])
    rejilla = cv2.resize((rejilla * 255).astype(np.uint8), None, fx=escala, fy=escala,
                         interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.applyColorMap(rejilla, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Analisis de una imagen
# --------------------------------------------------------------------------
def analizar(imagen, es_lienzo, varios, comparar, clave, pincel=None, ancho_lienzo=None):
    resultado = pre.procesar(
        imagen, varios=varios, es_lienzo=es_lienzo,
        fondo_transparente=0 if es_lienzo else 255,
    )
    if not resultado.digitos:
        st.warning("No encontré trazos en la imagen. Escribe el número más grande "
                   "y con buen contraste (tinta oscura sobre papel claro).")
        return

    prob = predecir(resultado.lote)
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    numero = "".join(str(d) for d in pred)

    col_res, col_img = st.columns([1, 1.4])
    with col_res:
        texto = "Número leído" if len(pred) > 1 else "Dígito"
        st.metric(texto, numero,
                  help="Cada dígito se separa, se lleva a 28×28 como en MNIST y se clasifica con la CNN.")
        st.caption(f"Confianza: {', '.join(f'{c:.0%}' for c in conf)}")
        dudosos = [i + 1 for i, c in enumerate(conf) if c < CONFIANZA_BAJA]
        if dudosos:
            st.warning(f"Poca confianza en el dígito {', '.join(map(str, dudosos))}. "
                       "Prueba a escribirlo más claro o más grande.")
        if pincel and ancho_lienzo:
            # Alto del digito mas pequeno, en pixeles del lienzo
            escala = resultado.gris.shape[1] / ancho_lienzo
            alto_min = min(d.caja[3] for d in resultado.digitos) / escala
            if alto_min < 3.5 * pincel:
                st.info(f"El número mide unos {alto_min:.0f} px y el pincel {pincel} px: con un trazo "
                        "tan grueso los huecos (0, 6, 8, 9) se tapan. Baja el grosor del pincel "
                        "para números pequeños.")
    with col_img:
        st.image(pre.dibujar_cajas(resultado, pred),
                 caption=f"{len(pred)} dígito(s) detectado(s)", width="stretch")

    st.markdown("##### Lo que recibe la red y lo que responde")
    for i in range(min(len(pred), MAX_DIGITOS_DETALLE)):
        c1, c2 = st.columns([1, 4], vertical_alignment="center")
        c1.image(ampliar(resultado.digitos[i].entrada), caption=f"#{i + 1} → {pred[i]}")
        c2.altair_chart(grafica_probabilidades(prob[i]), width="stretch")
    if len(pred) > MAX_DIGITOS_DETALLE:
        st.caption(f"Se muestran los primeros {MAX_DIGITOS_DETALLE} dígitos.")

    with st.expander("Pasos del preprocesamiento"):
        a, b, c = st.columns(3)
        a.image(resultado.gris, caption="1. Escala de grises", width="stretch")
        b.image(resultado.realce, width="stretch",
                caption="2. Trazos realzados" + (" (invertida: fondo claro)" if resultado.invertida else ""))
        c.image(resultado.mascara, caption="3. Umbral de Otsu y limpieza", width="stretch")
        st.caption("4. Cada dígito se recorta, se ajusta el grosor del trazo, se escala a 20×20 "
                   "y se centra por su centro de masa en 28×28, igual que las imágenes de MNIST.")

    with st.expander("Lo que ve la red: mapas de características"):
        elegido = 0
        if len(pred) > 1:
            elegido = st.selectbox("Dígito", range(len(pred)), key=f"mapa_{clave}",
                                   format_func=lambda i: f"#{i + 1} ({pred[i]})")
        _, visor = cargar_modelo()
        mapas1, mapas2 = visor(resultado.lote[elegido:elegido + 1], training=False)
        a, b = st.columns(2)
        a.image(mosaico(mapas1.numpy()[0], 8), width="stretch",
                caption="conv1: 32 filtros 3×3 (bordes y trazos simples), 26×26")
        b.image(mosaico(mapas2.numpy()[0], 8), width="stretch",
                caption="conv2: 64 filtros 3×3 (combinaciones de trazos), 13×13")

    if comparar:
        entrada = pre.preprocesar_basico(imagen if not es_lienzo else pre.a_gris(imagen, 0))
        p = predecir(entrada)[0]
        st.markdown("##### Enfoque básico (solo redimensionar a 28×28)")
        c1, c2 = st.columns([1, 4], vertical_alignment="center")
        c1.image(ampliar(entrada[0]), caption=f"→ {p.argmax()} ({p.max():.0%})")
        c2.altair_chart(grafica_probabilidades(p), width="stretch")
        st.caption("Sin invertir ni centrar, una foto de tinta oscura sobre papel no se parece a "
                   "MNIST y la red suele fallar; y solo puede leer un dígito por imagen.")


def abrir_imagen(archivo):
    """Archivo subido -> numpy RGB/RGBA respetando la orientacion EXIF del celular."""
    img = ImageOps.exif_transpose(Image.open(archivo))
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return np.asarray(img)


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
metricas = leer_json("metricas.json")

with st.sidebar:
    st.title("🔢 Dígitos con CNN")
    st.caption("Autor: Sebastian Lizcano")
    varios = st.toggle("Leer varios dígitos", value=True,
                       help="Separa los dígitos escritos en una fila y los lee de izquierda a derecha. "
                            "Apágalo para tratar todo lo dibujado como un solo dígito.")
    comparar = st.toggle("Comparar con el enfoque básico", value=False,
                         help="Muestra qué predice la red si la imagen solo se redimensiona a 28×28.")
    if metricas:
        st.divider()
        st.metric("Precisión en prueba", f"{metricas['precision_prueba']:.2%}",
                  help="10.000 imágenes de MNIST que la red nunca vio al entrenar.")
        st.caption(f"{metricas['parametros']:,} parámetros · {metricas['epocas_entrenadas']} épocas "
                   f"· TensorFlow {metricas['versiones']['tensorflow']}".replace(",", "."))

st.title("Reconocimiento de dígitos escritos a mano")
st.write("Una red neuronal convolucional entrenada con **MNIST** (60.000 dígitos manuscritos). "
         "Dibuja un número, sube una foto o usa la cámara.")

tab_dibujar, tab_subir, tab_camara, tab_modelo = st.tabs(
    ["✏️ Dibujar", "🖼️ Subir imagen", "📷 Cámara", "🧠 El modelo"])

with tab_dibujar:
    c1, c2 = st.columns([3, 1])
    grosor = c2.slider("Grosor del pincel", 3, 30, 6,
                       help="Con pincel delgado se reconocen números de cualquier tamaño, desde unos "
                            "20 px, en cualquier parte del recuadro. La app ajusta el grosor por su cuenta.")
    c2.caption("Dibuja donde quieras y del tamaño que quieras. "
               "Borra con el ícono de papelera de la barra del lienzo.")
    with c1:
        ancho = 560 if varios else 280
        lienzo = st_canvas(
            stroke_width=grosor, stroke_color="#FFFFFF", background_color="#000000",
            width=ancho, height=280, drawing_mode="freedraw", return_image_data=True,
            key=f"lienzo_{ancho}",
        )
    imagen_lienzo = lienzo.image_data
    if imagen_lienzo is not None and imagen_lienzo[..., :3].max() > 0:
        analizar(imagen_lienzo, es_lienzo=True, varios=varios, comparar=comparar, clave="lienzo",
                 pincel=grosor, ancho_lienzo=ancho)
    else:
        st.info("Dibuja un dígito" + (" o un número de varios dígitos" if varios else "")
                + " en el recuadro negro.")

with tab_subir:
    archivo = st.file_uploader("Imagen con un número (foto de un papel, captura, etc.)",
                               type=["png", "jpg", "jpeg", "webp", "bmp"])
    ejemplos = sorted(CARPETA_EJEMPLOS.glob("*.png"))
    ejemplo = st.selectbox("…o prueba con una imagen de ejemplo", [None] + ejemplos,
                           format_func=lambda p: "—" if p is None else p.stem.replace("_", " "))
    fuente = archivo or ejemplo
    if fuente is not None:
        analizar(abrir_imagen(fuente), es_lienzo=False, varios=varios, comparar=comparar,
                 clave="subir")

with tab_camara:
    st.caption("Consejo: escribe con marcador o lapicero oscuro sobre hoja blanca y "
               "acerca la cámara para que el número ocupe buena parte de la foto.")
    # La camara se activa a pedido: si no, el navegador pide permiso al abrir la app
    foto = None
    if st.toggle("Activar cámara", key="activar_camara"):
        foto = st.camera_input("Toma una foto del número")
    if foto is not None:
        analizar(abrir_imagen(foto), es_lienzo=False, varios=varios, comparar=comparar,
                 clave="camara")

with tab_modelo:
    historial = leer_json("historial.json")
    evaluacion = leer_json("evaluacion_preprocesamiento.json")
    if not metricas:
        st.error("Falta modelo/metricas.json. Ejecuta `python entrenar_modelo.py`.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precisión en prueba", f"{metricas['precision_prueba']:.2%}")
    m2.metric("Errores", f"{metricas['errores_prueba']} de 10.000")
    m3.metric("Parámetros", f"{metricas['parametros']:,}".replace(",", "."))
    m4.metric("Épocas", f"{metricas['epocas_entrenadas']}",
              help=f"Parada anticipada: los pesos guardados son los de la época {metricas['mejor_epoca']}.")

    st.subheader("Arquitectura")
    modelo, _ = cargar_modelo()
    capas = pd.DataFrame([{
        "Capa": capa.name,
        "Tipo": type(capa).__name__,
        "Salida": " × ".join(str(d) for d in capa.output.shape[1:]),
        "Parámetros": capa.count_params(),
    } for capa in modelo.layers])
    st.dataframe(capas, hide_index=True)
    st.caption("La del cuaderno de clase (Conv2D 32 → MaxPool → Conv2D 64 → MaxPool → Dense 64 → "
               "Dense 32 → Softmax 10) más un Dropout de 0,3. Durante el entrenamiento se aplicó "
               "aumentación de datos: rotación, traslación, zoom y grosor de trazo variable.")

    if historial:
        st.subheader("Entrenamiento")
        filas = []
        for serie, sufijo in (("Entrenamiento", ""), ("Validación", "val_")):
            for e, (acc, loss) in enumerate(zip(historial[f"{sufijo}accuracy"],
                                                historial[f"{sufijo}loss"]), start=1):
                filas.append({"Época": e, "Serie": serie, "Precisión": acc, "Pérdida": loss})
        df = pd.DataFrame(filas)
        colores = alt.Scale(domain=["Entrenamiento", "Validación"], range=[AZUL, NARANJA])

        def curva(campo, formato):
            return alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36)).encode(
                x=alt.X("Época:Q", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y(f"{campo}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=formato, title=None)),
                color=alt.Color("Serie:N", scale=colores, legend=alt.Legend(orient="top", title=None)),
                tooltip=["Serie:N", "Época:Q", alt.Tooltip(f"{campo}:Q", format=".4f")],
            ).properties(height=300, title=campo)

        a, b = st.columns(2)
        a.altair_chart(curva("Precisión", "%"), width="stretch")
        b.altair_chart(curva("Pérdida", ".2f"), width="stretch")
        st.caption("La validación usa 6.000 imágenes separadas del entrenamiento; el set de prueba "
                   "(10.000) solo se usó al final. La validación supera al entrenamiento porque las "
                   "imágenes de entrenamiento llevan aumentación y Dropout activo.")

    st.subheader("¿Con qué se confunde?")
    matriz = np.array(metricas["matriz_confusion"])
    celdas = pd.DataFrame([
        {"Real": str(r), "Predicho": str(p), "Imágenes": int(matriz[r, p])}
        for r in range(10) for p in range(10) if r != p
    ])
    base = alt.Chart(celdas).encode(
        x=alt.X("Predicho:N", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Real:N"),
        tooltip=["Real:N", "Predicho:N", "Imágenes:Q"],
    )
    # Todas las celdas en gris neutro y encima, en azul, solo las que tienen errores
    fondo = base.mark_rect(stroke=SUPERFICIE, strokeWidth=2, cornerRadius=2, color=NEUTRO)
    calor = base.transform_filter("datum['Imágenes'] > 0").mark_rect(
        stroke=SUPERFICIE, strokeWidth=2, cornerRadius=2).encode(
        color=alt.Color("Imágenes:Q", scale=alt.Scale(range=RAMPA_AZUL),
                        legend=alt.Legend(title="Errores")),
    )
    numeros = base.transform_filter("datum['Imágenes'] > 0").mark_text(fontSize=11).encode(
        text="Imágenes:Q",
        color=alt.condition(f"datum['Imágenes'] > {max(1, matriz[~np.eye(10, dtype=bool)].max() / 2)}",
                            alt.value(TEXTO_ALTO), alt.value(TEXTO_BAJO)),
    )
    c1, c2 = st.columns([1.2, 1])
    c1.altair_chart((fondo + calor + numeros).properties(height=340), width="stretch")
    c2.caption("Errores del set de prueba: fila = dígito real, columna = lo que dijo la red. "
               "La diagonal (aciertos) se omite para que se vean los errores.")
    errores = leer_errores()
    if errores is not None:
        c2.markdown("**Algunos errores** (real → predicho)")
        imgs = [cv2.resize(i, (56, 56), interpolation=cv2.INTER_NEAREST) for i in errores["imagenes"][:12]]
        c2.image(imgs, caption=[f"{r} → {p}" for r, p in zip(errores["real"], errores["predicho"])][:12])

    if evaluacion and "zonas_tamanos" in evaluacion:
        zt = evaluacion["zonas_tamanos"]
        st.subheader("¿Lo reconoce en cualquier parte y a cualquier tamaño?")
        st.write(f"Prueba con {zt['total']['imagenes']:,} dibujos simulados con pincel ".replace(",", ".")
                 + f"(dígitos del set de prueba): **{zt['total']['proyecto']:.1%}** de acierto con este "
                 f"proyecto frente a {zt['total']['basico']:.1%} con el enfoque básico (solo redimensionar).")

        def calor_precision(df, x, y, orden_x, orden_y, alto):
            base = alt.Chart(df).encode(
                x=alt.X(f"{x}:N", sort=orden_x, axis=alt.Axis(labelAngle=0, orient="top", title=None)),
                y=alt.Y(f"{y}:N", sort=orden_y, axis=alt.Axis(title=None)),
                tooltip=[f"{y}:N", f"{x}:N", alt.Tooltip("Acierto:Q", format=".1%"), "Imágenes:Q"],
            )
            celdas = base.mark_rect(cornerRadius=3, stroke=SUPERFICIE, strokeWidth=2).encode(
                color=alt.Color("Acierto:Q", legend=None,
                                scale=alt.Scale(domain=[0.5, 1], range=RAMPA_AZUL, clamp=True)))
            texto = base.mark_text(fontSize=13).encode(
                text=alt.Text("Acierto:Q", format=".0%"),
                color=alt.condition("datum['Acierto'] > 0.8", alt.value(TEXTO_ALTO), alt.value(TEXTO_BAJO)))
            return (celdas + texto).properties(height=alto)

        df_z = pd.DataFrame([{"Fila": z.split()[0].capitalize(), "Columna": z.split()[1].capitalize(),
                              "Acierto": v["proyecto"], "Imágenes": v["imagenes"]}
                             for z, v in zt["por_zona"].items()])
        filas_t = []
        for t, por_pincel in zt["tamano_x_pincel"].items():
            for pincel, v in por_pincel.items():
                if v:
                    filas_t.append({"Tamaño": t.capitalize(), "Pincel": f"{pincel} px",
                                    "Acierto": v["proyecto"], "Imágenes": v["imagenes"]})
        df_t = pd.DataFrame(filas_t)
        a, b = st.columns([1, 1.6])
        with a:
            st.markdown("**Por zona del lienzo** (promedio de todos los pinceles)")
            st.altair_chart(calor_precision(df_z, "Columna", "Fila", ["Izquierda", "Centro", "Derecha"],
                                            ["Arriba", "Centro", "Abajo"], 220), width="stretch")
        with b:
            st.markdown("**Por tamaño del número y grosor del pincel**")
            st.altair_chart(calor_precision(df_t, "Pincel", "Tamaño", list(dict.fromkeys(df_t["Pincel"])),
                                            [t.capitalize() for t in zt["tamano_x_pincel"]], 220),
                            width="stretch")
        st.caption("Tamaños (alto en un lienzo de 280 px): muy pequeño 20–36 px, pequeño 40–70, "
                   "mediano 80–150, grande 160–220, muy grande 230–280 (toca los bordes). "
                   "La posición no influye porque cada número se recorta y se centra antes de pasar "
                   "a la red; lo que sí influye es un pincel grueso en un número pequeño.")

    if evaluacion:
        st.subheader("Resultados con imágenes tipo app")
        def pct(v):
            return "no puede" if v is None else f"{v:.1%}".replace(".", ",")
        ev_l, ev_f, ev_n = evaluacion["lienzo"], evaluacion["foto"], evaluacion["numero"]
        tabla = pd.DataFrame([
            {"Escenario": "Dibujo en el lienzo (chico o grande, en cualquier parte)",
             "Enfoque básico": pct(ev_l["basico"]), "Este proyecto": pct(ev_l["proyecto"]),
             "Imágenes": ev_l["imagenes"]},
            {"Escenario": "Foto de papel (sombra, ruido, trazo fino)",
             "Enfoque básico": pct(ev_f["basico"]), "Este proyecto": pct(ev_f["proyecto"]),
             "Imágenes": ev_f["imagenes"]},
            {"Escenario": "Número de 4 dígitos en una foto (lectura exacta)",
             "Enfoque básico": pct(None), "Este proyecto": pct(ev_n["lectura_exacta"]),
             "Imágenes": ev_n["imagenes"]},
        ])
        st.dataframe(tabla, hide_index=True)
        st.caption("Imágenes generadas con dígitos del set de prueba de MNIST "
                   "(script `evaluar_preprocesamiento.py`). «Enfoque básico» = redimensionar la imagen "
                   "completa a 28×28; no puede leer varios dígitos.")
