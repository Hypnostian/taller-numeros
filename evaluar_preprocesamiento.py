"""Mide que tan bien se reconocen digitos en imagenes parecidas a las de la app.

Toma digitos del set de PRUEBA de MNIST (no usados en el entrenamiento) y los
convierte en tres tipos de imagen:

  * lienzo: trazo blanco sobre negro, digito chico o grande y fuera del centro
    (como cuando alguien dibuja en la app).
  * foto: tinta oscura sobre papel, con sombra, ruido, desenfoque, trazo mas
    fino y algo de rotacion (como una foto del cuaderno).
  * numero: cuatro digitos seguidos en una foto (lectura de varios digitos).
  * zonas y tamanos: digitos "dibujados" con un pincel de grosor fijo en las
    9 zonas del lienzo (esquinas incluidas) y en 5 tamanos, de 20 px a 280 px.

Compara el enfoque basico (redimensionar toda la imagen a 28x28) con el
preprocesamiento de este proyecto. Guarda modelo/evaluacion_preprocesamiento.json
y algunas imagenes de muestra en ejemplos/.

Uso:
    python evaluar_preprocesamiento.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import preprocesamiento as pre

RAIZ = Path(__file__).parent
SEMILLA = 7
N_LIENZO = 1000
N_FOTO = 1000
N_NUMEROS = 250  # de 4 digitos cada uno


def digito_grande(img28, alto, rng, adelgazar=False):
    """Digito MNIST recortado y ampliado a `alto` pixeles, como mascara 0-255."""
    binaria = np.where(img28 > 60, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(binaria)
    recorte = img28[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = alto / recorte.shape[0]
    ancho = max(2, round(recorte.shape[1] * f))
    grande = cv2.resize(recorte, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    grande = cv2.GaussianBlur(grande, (5, 5), 0)
    grande = np.where(grande > 110, 255, 0).astype(np.uint8)  # trazo solido
    if adelgazar:  # un lapicero deja un trazo mas fino que el de MNIST
        k = int(rng.integers(1, max(2, alto // 25)))
        if k > 1:
            delgada = cv2.erode(grande, np.ones((k, k), np.uint8))
            if np.count_nonzero(delgada) > 0.3 * np.count_nonzero(grande):
                grande = delgada
    return grande


def rotar(mascara, grados):
    h, w = mascara.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(mascara, m, (nw, nh), borderValue=0)


def hacer_lienzo(img28, rng):
    lienzo = np.zeros((280, 280), np.uint8)
    d = digito_grande(img28, int(rng.integers(70, 241)), rng)
    h, w = d.shape
    if w > 270:
        return None
    y = int(rng.integers(0, 280 - h + 1))
    x = int(rng.integers(0, 280 - w + 1))
    lienzo[y:y + h, x:x + w] = np.maximum(lienzo[y:y + h, x:x + w], d)
    return lienzo


def papel(alto, ancho, rng):
    """Hoja con sombra en una direccion aleatoria."""
    base = rng.uniform(170, 235)
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    rampa = np.cos(ang) * xx / ancho + np.sin(ang) * yy / alto
    rampa = (rampa - rampa.min()) / (np.ptp(rampa) + 1e-6)
    sombra = 1.0 - rng.uniform(0.15, 0.45) * rampa
    return base * sombra


def entintar(fondo, mascara, rng):
    """Pone tinta oscura donde la mascara es blanca y agrega ruido y desenfoque."""
    alfa = cv2.GaussianBlur(mascara.astype(np.float32) / 255.0, (3, 3), 0)
    tinta = rng.uniform(15, 80)
    img = fondo * (1 - alfa) + tinta * alfa
    img += rng.normal(0, rng.uniform(3, 10), img.shape)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def hacer_foto(img28, rng):
    alto, ancho = 480, 640
    d = rotar(digito_grande(img28, int(rng.integers(110, 301)), rng, adelgazar=True),
              rng.uniform(-10, 10))
    h, w = d.shape
    if h > alto - 20 or w > ancho - 20:
        return None
    mascara = np.zeros((alto, ancho), np.uint8)
    y = int(rng.integers(10, alto - h - 9))
    x = int(rng.integers(10, ancho - w - 9))
    mascara[y:y + h, x:x + w] = d
    return entintar(papel(alto, ancho, rng), mascara, rng)


def hacer_numero(imgs28, rng):
    alto_d = int(rng.integers(110, 181))
    piezas = [rotar(digito_grande(i, alto_d, rng, adelgazar=True), rng.uniform(-6, 6))
              for i in imgs28]
    huecos = [int(rng.integers(25, 70)) for _ in piezas]
    ancho = sum(p.shape[1] for p in piezas) + sum(huecos) + 80
    alto = max(p.shape[0] for p in piezas) + 120
    mascara = np.zeros((alto, ancho), np.uint8)
    x = 40
    for p, g in zip(piezas, huecos):
        y = (alto - p.shape[0]) // 2 + int(rng.integers(-15, 16))
        mascara[y:y + p.shape[0], x:x + p.shape[1]] = np.maximum(
            mascara[y:y + p.shape[0], x:x + p.shape[1]], p)
        x += p.shape[1] + g
    return entintar(papel(alto, ancho, rng), mascara, rng)


# --- Dibujo con pincel en cualquier zona y tamano --------------------------------
TAMANOS = {  # alto total del digito (pixeles) en un lienzo de 280x280
    "muy pequeño": (20, 36),
    "pequeño": (40, 70),
    "mediano": (80, 150),
    "grande": (160, 220),
    "muy grande": (230, 280),
}
PINCELES = (4, 6, 10, 16, 24, 32)
ZONAS = [f"{v} {h}" for v in ("arriba", "centro", "abajo") for h in ("izquierda", "centro", "derecha")]


def esqueleto_mnist(img28, alto=300):
    """Recorrido del lapiz de un digito MNIST (esqueleto de 1 px) a `alto` px."""
    binaria = np.where(img28 > 60, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(binaria)
    recorte = img28[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = alto / recorte.shape[0]
    grande = cv2.resize(recorte, (max(2, round(recorte.shape[1] * f)), alto),
                        interpolation=cv2.INTER_CUBIC)
    grande = np.where(cv2.GaussianBlur(grande, (9, 9), 0) > 110, 255, 0).astype(np.uint8)
    grande = cv2.copyMakeBorder(grande, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    return cv2.ximgproc.thinning(grande)


def dibujar_con_pincel(esqueleto, alto_total, pincel):
    """Simula a alguien dibujando el digito en la app: el pincel tiene un
    grosor fijo en pixeles sin importar lo grande o chico que sea el numero."""
    alto = alto_total - pincel
    if alto < 6:
        return None
    ys, xs = np.nonzero(esqueleto)
    esqueleto = esqueleto[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = alto / esqueleto.shape[0]
    ancho = max(1, round(esqueleto.shape[1] * f))
    linea = cv2.resize(esqueleto, (ancho, alto), interpolation=cv2.INTER_AREA)
    linea = np.where(linea > 0, 255, 0).astype(np.uint8)
    linea = cv2.copyMakeBorder(linea, pincel, pincel, pincel, pincel, cv2.BORDER_CONSTANT, value=0)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pincel, pincel))
    trazo = cv2.dilate(linea, nucleo)
    ys, xs = np.nonzero(trazo)
    return trazo[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def colocar(digito, zona, rng, lado=280):
    """Pone el digito en una de las 9 zonas del lienzo (pegado o casi al borde)."""
    h, w = digito.shape
    if h > lado or w > lado:
        return None
    fila, col = divmod(zona, 3)
    def pos(i, tam):
        libre = lado - tam
        return [int(rng.integers(0, min(8, libre) + 1)), libre // 2,
                libre - int(rng.integers(0, min(8, libre) + 1))][i]
    y, x = pos(fila, h), pos(col, w)
    lienzo = np.zeros((lado, lado), np.uint8)
    lienzo[y:y + h, x:x + w] = digito
    return lienzo


def evaluar_posiciones_y_tamanos(modelo, x_pru, y_pru, indices, rng, por_zona=20):
    """Precision por tamano x pincel y por zona, frente al enfoque basico."""
    muestras = []  # (imagen, etiqueta, tamano, pincel, zona)
    k = 0
    esqueletos = {}
    for tamano, (lo, hi) in TAMANOS.items():
        for pincel in PINCELES:
            for zona in range(9):
                for _ in range(por_zona):
                    i = indices[k % len(indices)]
                    k += 1
                    if i not in esqueletos:
                        esqueletos[i] = esqueleto_mnist(x_pru[i])
                    trazo = dibujar_con_pincel(esqueletos[i], int(rng.integers(lo, hi + 1)), pincel)
                    img = None if trazo is None else colocar(trazo, zona, rng)
                    if img is not None:
                        muestras.append((img, y_pru[i], tamano, pincel, zona))

    etiquetas = np.array([m[1] for m in muestras])
    basico = predecir(modelo, [pre.preprocesar_basico(m[0])[0] for m in muestras])
    propias = []
    for m in muestras:
        r = pre.procesar(m[0], varios=False, es_lienzo=True)
        propias.append(r.lote[0] if r.digitos else np.zeros((28, 28, 1), np.float32))
    propio = predecir(modelo, propias)

    def resumen(filtro):
        sel = np.array([filtro(m) for m in muestras])
        if not sel.any():
            return None
        return {"imagenes": int(sel.sum()),
                "basico": float((basico[sel] == etiquetas[sel]).mean()),
                "proyecto": float((propio[sel] == etiquetas[sel]).mean())}

    return {
        "total": resumen(lambda m: True),
        "por_tamano": {t: resumen(lambda m, t=t: m[2] == t) for t in TAMANOS},
        "por_zona": {ZONAS[z]: resumen(lambda m, z=z: m[4] == z) for z in range(9)},
        "tamano_x_pincel": {t: {str(p): resumen(lambda m, t=t, p=p: m[2] == t and m[3] == p)
                                for p in PINCELES} for t in TAMANOS},
    }


def predecir(modelo, lote):
    if len(lote) == 0:
        return np.zeros(0, int)
    return modelo.predict(np.asarray(lote, np.float32), batch_size=256, verbose=0).argmax(1)


def evaluar_un_digito(modelo, imagenes, etiquetas, es_lienzo):
    basico, propio, vacios = [], [], 0
    for img in imagenes:
        basico.append(pre.preprocesar_basico(img)[0])
        r = pre.procesar(img, varios=False, es_lienzo=es_lienzo)
        if r.digitos:
            propio.append(r.lote[0])
        else:  # no encontro nada: cuenta como error
            propio.append(np.zeros((28, 28, 1), np.float32))
            vacios += 1
    acierto_basico = float((predecir(modelo, basico) == etiquetas).mean())
    acierto_propio = float((predecir(modelo, propio) == etiquetas).mean())
    return acierto_basico, acierto_propio, vacios


def main():
    tf.keras.utils.set_random_seed(SEMILLA)
    rng = np.random.default_rng(SEMILLA)
    modelo = tf.keras.models.load_model(RAIZ / "modelo" / "numeros_cnn.keras")
    _, (x_pru, y_pru) = tf.keras.datasets.mnist.load_data()
    orden = rng.permutation(len(x_pru))

    resultados = {}
    carpeta_ejemplos = RAIZ / "ejemplos"
    carpeta_ejemplos.mkdir(exist_ok=True)

    # --- Lienzo ---------------------------------------------------------------
    imgs, etqs = [], []
    for i in orden[:N_LIENZO]:
        img = hacer_lienzo(x_pru[i], rng)
        if img is not None:
            imgs.append(img)
            etqs.append(y_pru[i])
    c, p, v = evaluar_un_digito(modelo, imgs, np.array(etqs), es_lienzo=True)
    resultados["lienzo"] = {"imagenes": len(imgs), "basico": c, "proyecto": p, "sin_detectar": v}
    print(f"Lienzo  ({len(imgs)}): basico {c:.1%}  proyecto {p:.1%}")

    # --- Foto de un digito -----------------------------------------------------
    imgs, etqs = [], []
    for i in orden[N_LIENZO:N_LIENZO + N_FOTO]:
        img = hacer_foto(x_pru[i], rng)
        if img is not None:
            imgs.append(img)
            etqs.append(y_pru[i])
    c, p, v = evaluar_un_digito(modelo, imgs, np.array(etqs), es_lienzo=False)
    resultados["foto"] = {"imagenes": len(imgs), "basico": c, "proyecto": p, "sin_detectar": v}
    print(f"Foto    ({len(imgs)}): basico {c:.1%}  proyecto {p:.1%}")
    for k in range(3):
        cv2.imwrite(str(carpeta_ejemplos / f"foto_{etqs[k]}.png"), imgs[k])

    # --- Numero de 4 digitos ---------------------------------------------------
    inicio = N_LIENZO + N_FOTO
    exactos, conteo_ok, digitos_ok, digitos_total = 0, 0, 0, 0
    for n in range(N_NUMEROS):
        idx = orden[inicio + 4 * n: inicio + 4 * n + 4]
        img = hacer_numero(x_pru[idx], rng)
        verdad = "".join(str(d) for d in y_pru[idx])
        r = pre.procesar(img, varios=True)
        leido = "".join(str(d) for d in predecir(modelo, r.lote))
        digitos_total += 4
        if len(leido) == 4:
            conteo_ok += 1
            digitos_ok += sum(a == b for a, b in zip(leido, verdad))
        exactos += leido == verdad
        if n < 2:
            cv2.imwrite(str(carpeta_ejemplos / f"numero_{verdad}.png"), img)
    resultados["numero"] = {
        "imagenes": N_NUMEROS,
        "lectura_exacta": exactos / N_NUMEROS,
        "conteo_correcto": conteo_ok / N_NUMEROS,
        "digitos_correctos": digitos_ok / digitos_total,
    }
    print(f"Numero  ({N_NUMEROS}): lectura exacta {exactos / N_NUMEROS:.1%}  "
          f"cantidad de digitos correcta {conteo_ok / N_NUMEROS:.1%}  "
          f"digitos correctos {digitos_ok / digitos_total:.1%}")

    # --- Zonas y tamanos con pincel ----------------------------------------------
    zonas = evaluar_posiciones_y_tamanos(modelo, x_pru, y_pru, orden[inicio + 4 * N_NUMEROS:], rng)
    resultados["zonas_tamanos"] = zonas
    print(f"Zonas y tamanos ({zonas['total']['imagenes']}): basico {zonas['total']['basico']:.1%}  "
          f"proyecto {zonas['total']['proyecto']:.1%}")
    for t, d in zonas["tamano_x_pincel"].items():
        print(f"  {t:12s}", "  ".join(f"pincel {p}: {v['proyecto']:.0%}" for p, v in d.items() if v))

    # Un dibujo de lienzo pequeno y descentrado para probar sin dibujar
    for i in orden[::-1]:
        img = hacer_lienzo(x_pru[i], rng)
        if img is not None:
            cv2.imwrite(str(carpeta_ejemplos / f"lienzo_{y_pru[i]}.png"), img)
            break

    (RAIZ / "modelo" / "evaluacion_preprocesamiento.json").write_text(
        json.dumps(resultados, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
