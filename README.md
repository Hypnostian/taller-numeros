# Reconocimiento de dígitos escritos a mano con una CNN

**Autor:** Sebastian Lizcano

Aplicación de Streamlit que reconoce números escritos a mano con una red neuronal
convolucional (CNN) entrenada con MNIST. Se puede **dibujar** el número en un lienzo,
**subir una foto** de un papel o usar la **cámara**. Lee un dígito o un número de
varios dígitos escrito en una fila, **en cualquier parte de la imagen y a cualquier
tamaño**.

Implementa el taller *Cuaderno 8 A2: CNN para reconocimiento de dígitos (MNIST)*:
cargar MNIST, normalizar, crear la CNN (Conv2D + MaxPooling + Dense), entrenar con
parada anticipada, guardar el modelo en `.keras` (y la arquitectura en JSON y los
pesos en `.weights.h5`), preprocesar una imagen nueva y predecir el dígito.

## Qué hace la app

| Pestaña | Qué hace |
|---|---|
| ✏️ Dibujar | Lienzo negro; dibuja uno o varios dígitos donde quieras y del tamaño que quieras. Predice en vivo al soltar el trazo. |
| 🖼️ Subir imagen | Foto o captura con un número (PNG, JPG, WEBP, BMP), o una de las imágenes de `ejemplos/`. Respeta la orientación de las fotos del celular (EXIF). |
| 📷 Cámara | Toma una foto del número escrito en papel (la cámara se activa con un interruptor, para no pedir permiso al abrir la app). |
| 🧠 El modelo | Precisión en prueba, arquitectura, curvas de entrenamiento, matriz de confusión, ejemplos de errores y resultados medidos. |

En cada predicción se muestra:

- El número leído, la confianza de cada dígito y la imagen con una caja sobre cada dígito.
- La entrada 28×28 que recibe la red y la probabilidad de cada clase (0–9).
- **Pasos del preprocesamiento** (gris → trazos realzados → umbral).
- **Mapas de características** de las dos capas convolucionales: lo que "ve" la red.
- Avisos cuando la confianza es baja o el pincel es muy grueso para un número pequeño.
- Opcional: comparación con el enfoque básico (solo redimensionar la imagen a 28×28).

## Base: MNIST y la red convolucional

El conjunto de datos es **MNIST** (LeCun, Cortes y Burges): 70.000 dígitos manuscritos
de 28×28 píxeles. Aquí se usan 54.000 para entrenar, 6.000 para validar y las 10.000
de prueba **solo al final**, para que la precisión reportada sea honesta.

| Capa | Salida | Parámetros |
|---|---|---|
| Conv2D 32 filtros 3×3, ReLU | 26 × 26 × 32 | 320 |
| MaxPooling 2×2 | 13 × 13 × 32 | 0 |
| Conv2D 64 filtros 3×3, ReLU | 13 × 13 × 64 | 18.496 |
| MaxPooling 2×2 | 6 × 6 × 64 | 0 |
| Flatten + Dropout 0,3 | 2.304 | 0 |
| Dense 64, ReLU | 64 | 147.520 |
| Dense 32, ReLU | 32 | 2.080 |
| Dense 10, Softmax | 10 | 330 |
| **Total** | | **168.746** |

Entrenamiento: Adam, `SparseCategoricalCrossentropy`, lotes de 32, `EarlyStopping`
(paciencia 5, restaura los mejores pesos) y `ReduceLROnPlateau`. 29 épocas en unos
6 minutos en CPU (TensorFlow 2.21, Keras 3.15).

## Aportes de este proyecto

1. **Detecta el número en cualquier parte y a cualquier tamaño.** La CNN espera el
   dígito centrado en 28×28, como en MNIST. En vez de redimensionar la imagen completa
   (con lo que un número chico en una esquina queda como unos pocos píxeles), cada
   dígito se localiza, se recorta, se escala a 20×20 y se centra por su centro de masa,
   que es como se construyó MNIST. Así da igual si se dibuja arriba a la izquierda,
   abajo a la derecha, muy pequeño o tocando los bordes.
2. **Normaliza el grosor del trazo.** Mide el grosor con la transformada de distancia y
   engrosa trazos finos (lapicero) o adelgaza trazos gruesos (marcador) hasta el grosor
   típico de MNIST, medido sobre el propio conjunto de datos.
3. **Funciona con fotos reales.** Invierte automáticamente si el fondo es claro, corrige
   sombras con un *top-hat* morfológico, binariza con Otsu y elimina motas y bordes de
   la hoja.
4. **Lee números de varios dígitos** escritos en una fila (componentes conectados
   agrupados por columnas, para que un 5 o un 4 con trazos separados cuenten como un
   solo dígito).
5. **Modelo extendido con aumentación de datos:** rotación (±14°), traslación (±10 %),
   zoom (±15 %) y **grosor de trazo variable** (dilatación aleatoria con *max pooling*
   3×3), más Dropout. Lo hace más tolerante a pinceles gruesos sin perder precisión en
   MNIST.
6. **Explicable y medible:** pasos del preprocesamiento, mapas de características,
   matriz de confusión, galería de errores y un script de evaluación reproducible.

## Resultados medidos

**MNIST, 10.000 imágenes de prueba nunca vistas:** 99,34 % de precisión (66 errores).

**Imágenes tipo app** (generadas con `evaluar_preprocesamiento.py` a partir de dígitos
del set de prueba). El enfoque básico es redimensionar la imagen completa a 28×28:

| Escenario | Enfoque básico | Este proyecto | Imágenes |
|---|---|---|---|
| Dibujo en el lienzo (chico o grande, descentrado) | 59,6 % | **99,4 %** | 990 |
| Foto de papel (sombra, ruido, trazo fino, rotación) | 12,0 % | **98,8 %** | 1.000 |
| Número de 4 dígitos en una foto (lectura exacta) | no puede | **95,6 %** | 250 |

**Zonas y tamaños:** 4.997 dibujos simulados con un pincel de grosor fijo (como en la
app) en las 9 zonas del lienzo de 280 px. Por zona el acierto va de 86 % a 90 % (es el
promedio de todos los pinceles, incluidos los muy gruesos): **la posición no influye**.
Lo que influye es el grosor del pincel frente al tamaño del número:

| Tamaño (alto) | Pincel 4 px | 6 px (por defecto) | 10 px | 16 px | 24 px | 32 px | Enfoque básico |
|---|---|---|---|---|---|---|---|
| Muy pequeño (20–36 px) | 97 % | 94 % | 70 % | 22 % | 8 % | — | 10 % |
| Pequeño (40–70 px) | 99 % | 98 % | 97 % | 92 % | 54 % | 19 % | 11 % |
| Mediano (80–150 px) | 98 % | 97 % | 96 % | 99 % | 98 % | 92 % | 22 % |
| Grande (160–220 px) | 99 % | 97 % | 98 % | 97 % | 97 % | 98 % | 58 % |
| Muy grande (230–280 px) | 99 % | 98 % | 98 % | 99 % | 100 % | 98 % | 71 % |

Con un pincel grueso en un número muy pequeño los huecos del 0, 6, 8 y 9 se tapan y el
dígito queda como una mancha; por eso la app usa un pincel delgado por defecto y avisa
cuando el número mide menos de 3,5 veces el grosor del pincel.

Durante el desarrollo, la aumentación de grosor (junto con un zoom algo mayor) mejoró
justamente esos casos frente a la misma red entrenada sin ella: pequeño con pincel de 16 px de 74 % a 93 %, mediano con 32 px
de 74 % a 92 % y muy pequeño con 10 px de 51 % a 73 %, con la precisión en MNIST
prácticamente igual (99,39 % → 99,34 %).

## Estructura

```
taller-numeros/
├── app.py                        # App de Streamlit
├── preprocesamiento.py           # Imagen -> entradas 28x28 tipo MNIST
├── entrenar_modelo.py            # Entrena la CNN y guarda modelo y métricas
├── evaluar_preprocesamiento.py   # Mide el acierto con imágenes tipo app
├── requirements.txt
├── ejemplos/                     # Imágenes para probar sin dibujar
└── modelo/
    ├── numeros_cnn.keras         # Modelo que usa la app
    ├── numeros_cnn.json          # Arquitectura (como en el cuaderno)
    ├── numeros_cnn.weights.h5    # Pesos (como en el cuaderno)
    ├── historial.json            # Precisión y pérdida por época
    ├── metricas.json             # Precisión en prueba, matriz de confusión...
    ├── errores_prueba.npz        # Algunos dígitos mal clasificados
    └── evaluacion_preprocesamiento.json
```

## Ejecutar en local (Windows)

Requiere Python 3.10 a 3.13 (TensorFlow 2.21 no tiene versión para 3.14).

```bash
py -3.11 -m venv .venv
```

```bash
.venv\Scripts\pip install -r requirements.txt
```

```bash
.venv\Scripts\streamlit run app.py
```

Para volver a entrenar y a medir (opcional, el modelo ya viene entrenado):

```bash
.venv\Scripts\python entrenar_modelo.py
```

```bash
.venv\Scripts\python evaluar_preprocesamiento.py
```

## Desplegar en Streamlit Community Cloud

1. Sube la carpeta `taller-numeros/` a un repositorio de GitHub, **incluyendo**
   `modelo/numeros_cnn.keras`, `modelo/*.json`, `modelo/errores_prueba.npz` y `ejemplos/`
   (el `.gitignore` solo excluye el entorno virtual y la caché).
2. Entra a [share.streamlit.io](https://share.streamlit.io) → **Create app** → elige el
   repositorio y la rama.
3. En **Main file path** escribe la ruta del archivo, por ejemplo `taller-numeros/app.py`.
   Streamlit busca el `requirements.txt` primero en esa misma carpeta.
4. En **Advanced settings** elige **Python 3.12** (o 3.11). No uses 3.14.
5. **Deploy**. La primera instalación tarda unos minutos por TensorFlow; después la app
   carga el modelo una sola vez y lo mantiene en memoria.

## Limitaciones y próximas mejoras

- Varios dígitos se leen en **una sola fila**; dígitos que se tocan se leen como uno.
- En fotos, las rayas o la cuadrícula del cuaderno pueden aparecer como trazos; funciona
  mejor en hoja blanca.
- Un número muy pequeño con pincel grueso pierde sus huecos (la app avisa).
- Solo reconoce dígitos del 0 al 9; cualquier otro trazo se clasifica como el dígito más
  parecido (la confianza baja lo delata).
- Mejoras posibles: separar dígitos que se tocan, leer varias filas y entrenar con
  EMNIST para incluir letras.
