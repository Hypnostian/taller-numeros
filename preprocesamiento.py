"""Convierte una imagen cualquiera (lienzo, foto o archivo) en entradas tipo MNIST.

MNIST tiene digitos blancos sobre fondo negro, escalados para caber en una caja
de 20x20 y centrados por su centro de masa dentro de una imagen de 28x28.
Una foto de un cuaderno es casi lo contrario: tinta oscura sobre papel claro,
con sombras, ruido y el numero en cualquier parte. Por eso redimensionar la
imagen completa a 28x28 (el enfoque basico) falla con imagenes reales.

Pasos de este modulo:
  1. Gris y tamano de trabajo.
  2. Polaridad: si el fondo es claro se invierte (tinta -> blanco).
  3. Quitar iluminacion desigual con un "top-hat" (fondo estimado por apertura).
  4. Umbral de Otsu y limpieza de motas.
  5. Segmentar los digitos (componentes conectados agrupados por columnas).
  6. Cada digito: normalizar grosor del trazo, escalar a 20x20, centrar por
     centro de masa en 28x28 y llevar a [0, 1].
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# Tamano de trabajo: imagenes grandes se reducen (velocidad) y las muy
# pequenas se amplian (para que la limpieza morfologica no borre trazos).
LADO_MIN = 280
LADO_MAX = 1000

# Escala canonica donde se mide y corrige el grosor del trazo.
LADO_CANONICO = 200

# Grosor del trazo relativo al lado mayor del digito, medido sobre el set de
# entrenamiento de MNIST con medir_grosor_mnist() (mediana).
GROSOR_RELATIVO_MNIST = 0.12

# Contraste minimo (0-255) entre trazo y fondo para considerar que hay algo.
CONTRASTE_MINIMO = 35


@dataclass
class Digito:
    entrada: np.ndarray  # 28x28 float32 en [0, 1], listo para la red
    caja: tuple  # (x, y, ancho, alto) en la imagen de trabajo


@dataclass
class Resultado:
    gris: np.ndarray  # imagen de trabajo en grises
    realce: np.ndarray  # trazos claros sobre fondo negro, sin sombras
    mascara: np.ndarray  # binaria despues de limpiar ruido
    invertida: bool  # True si el fondo era claro (tinta oscura)
    digitos: list = field(default_factory=list)

    @property
    def lote(self):
        """Todas las entradas como lote (n, 28, 28, 1) para model.predict."""
        if not self.digitos:
            return np.zeros((0, 28, 28, 1), np.float32)
        return np.stack([d.entrada for d in self.digitos])[..., None]


def a_gris(imagen, fondo_transparente=255):
    """Acepta gris, RGB o RGBA (uint8) y devuelve gris uint8.

    Las zonas transparentes se rellenan con `fondo_transparente`
    (blanco por defecto, como si fuera papel).
    """
    imagen = np.asarray(imagen)
    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)
    if imagen.ndim == 2:
        return imagen
    if imagen.shape[2] == 4:
        rgb = imagen[..., :3].astype(np.float32)
        alfa = imagen[..., 3:4].astype(np.float32) / 255.0
        rgb = rgb * alfa + fondo_transparente * (1.0 - alfa)
        imagen = rgb.astype(np.uint8)
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)


def _tamano_de_trabajo(gris):
    h, w = gris.shape
    lado = max(h, w)
    if lado > LADO_MAX:
        f = LADO_MAX / lado
        return cv2.resize(gris, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if lado < LADO_MIN:
        f = LADO_MIN / lado
        return cv2.resize(gris, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return gris


def realzar_trazos(gris, corregir_fondo=True):
    """Devuelve (realce, invertida, contraste).

    realce: trazos claros sobre fondo negro, con la iluminacion corregida.
    corregir_fondo=False omite la correccion de sombras (el lienzo no tiene
    sombras, y asi un digito enorme con trazo muy grueso no se borra).
    """
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    invertida = bool(np.median(suave) > 127)
    if invertida:
        suave = 255 - suave

    # El fondo se estima con una apertura morfologica con un nucleo mas ancho
    # que cualquier trazo: borra los trazos y deja sombras y gradientes.
    # Se calcula en pequeno porque un nucleo grande a resolucion completa es lento.
    if corregir_fondo:
        h, w = suave.shape
        f = min(1.0, 200 / max(h, w))
        peq = cv2.resize(suave, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        k = max(3, min(peq.shape) // 5) | 1
        nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        fondo = cv2.morphologyEx(peq, cv2.MORPH_OPEN, nucleo)
        fondo = cv2.resize(fondo, (w, h), interpolation=cv2.INTER_LINEAR)
        realce = cv2.subtract(suave, fondo)
    else:
        realce = suave
    contraste = int(realce.max())
    if contraste > 0:
        realce = cv2.convertScaleAbs(realce, alpha=255.0 / contraste)
    return realce, invertida, contraste


def _mascara_limpia(realce):
    t, _ = cv2.threshold(realce, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mascara = cv2.threshold(realce, max(t, 50), 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _componentes_validos(mascara, es_lienzo):
    """Etiqueta componentes y descarta motas y bordes de papel/mesa."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    H, W = mascara.shape
    candidatos = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        toca_borde = x == 0 or y == 0 or x + w >= W or y + h >= H
        # En fotos, una franja que cruza casi toda la imagen pegada al borde
        # es el borde del papel o la mesa, no un digito.
        if not es_lienzo and toca_borde and (w > 0.9 * W or h > 0.9 * H):
            continue
        candidatos.append((i, x, y, w, h, area))
    if not candidatos:
        return etiquetas, []
    area_max = max(c[5] for c in candidatos)
    return etiquetas, [c for c in candidatos if c[5] >= 0.04 * area_max]


def _agrupar_por_columnas(componentes):
    """Une componentes que se solapan en horizontal (p. ej. un 5 con la raya
    de arriba separada) y devuelve grupos ordenados de izquierda a derecha."""
    grupos = [{"ids": [c[0]], "x1": c[1], "y1": c[2], "x2": c[1] + c[3], "y2": c[2] + c[4]}
              for c in componentes]
    unido = True
    while unido:
        unido = False
        for a in range(len(grupos)):
            for b in range(a + 1, len(grupos)):
                ga, gb = grupos[a], grupos[b]
                solape = min(ga["x2"], gb["x2"]) - max(ga["x1"], gb["x1"])
                ancho_menor = min(ga["x2"] - ga["x1"], gb["x2"] - gb["x1"])
                if solape > 0.5 * ancho_menor:
                    ga["ids"] += gb["ids"]
                    for k, fn in (("x1", min), ("y1", min), ("x2", max), ("y2", max)):
                        ga[k] = fn(ga[k], gb[k])
                    del grupos[b]
                    unido = True
                    break
            if unido:
                break
    return sorted(grupos, key=lambda g: g["x1"])


def _un_solo_grupo(componentes):
    return [{
        "ids": [c[0] for c in componentes],
        "x1": min(c[1] for c in componentes),
        "y1": min(c[2] for c in componentes),
        "x2": max(c[1] + c[3] for c in componentes),
        "y2": max(c[2] + c[4] for c in componentes),
    }]


def _grosor(binaria):
    """Ancho medio del trazo en pixeles: en un trazo de ancho g la distancia
    al borde va de 0 a g/2, asi que su media es g/4 (menos el sesgo discreto)."""
    # Borde negro de 1 px: si el recorte esta lleno (un "1" recto y grueso) no
    # habria fondo y la distancia saldria infinita.
    binaria = cv2.copyMakeBorder(binaria, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(binaria, cv2.DIST_L2, 3)
    valores = dist[binaria > 0]
    if valores.size == 0:
        return 0.0
    return max(4.0 * float(valores.mean()) - 2.0, 1.0)


def _a_escala_canonica(binaria):
    h, w = binaria.shape
    f = LADO_CANONICO / max(h, w)
    nuevo = (max(1, round(w * f)), max(1, round(h * f)))
    reescalada = cv2.resize(binaria, nuevo, interpolation=cv2.INTER_LINEAR)
    return np.where(reescalada > 127, 255, 0).astype(np.uint8)


def _recortar(binaria):
    ys, xs = np.nonzero(binaria)
    if ys.size == 0:
        return binaria
    return binaria[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalizar_grosor(binaria):
    """Engrosa trazos finos (lapicero) o adelgaza trazos gruesos (marcador)
    hasta el grosor relativo tipico de MNIST. Trabaja a escala canonica."""
    binaria = _a_escala_canonica(_recortar(binaria))
    objetivo = GROSOR_RELATIVO_MNIST * LADO_CANONICO
    actual = _grosor(binaria)
    margen = int(objetivo) + 2
    con_margen = cv2.copyMakeBorder(binaria, margen, margen, margen, margen,
                                    cv2.BORDER_CONSTANT, value=0)

    if actual < 0.8 * objetivo:
        k = int(round(objetivo - actual)) + 1
        nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return _recortar(cv2.dilate(con_margen, nucleo))

    if actual > 1.5 * objetivo:
        k = int(round(actual - objetivo)) + 1
        nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        erosionada = cv2.erode(con_margen, nucleo)
        # Si la erosion se come el digito (trazos muy desiguales) no se aplica.
        if np.count_nonzero(erosionada) > 0.4 * np.count_nonzero(con_margen):
            return _recortar(erosionada)
    return binaria


def a_formato_mnist(binaria):
    """Recorte binario de un digito -> 28x28 float32 en [0, 1] al estilo MNIST."""
    digito = normalizar_grosor(binaria)
    h, w = digito.shape
    f = 20.0 / max(h, w)
    nuevo = (max(1, round(w * f)), max(1, round(h * f)))
    # INTER_AREA promedia pixeles: deja bordes suaves como los de MNIST.
    pequeno = cv2.resize(digito, nuevo, interpolation=cv2.INTER_AREA)

    lienzo = np.zeros((28, 28), np.uint8)
    y0 = (28 - nuevo[1]) // 2
    x0 = (28 - nuevo[0]) // 2
    lienzo[y0:y0 + nuevo[1], x0:x0 + nuevo[0]] = pequeno

    # MNIST centra cada digito por su centro de masa.
    m = cv2.moments(lienzo)
    if m["m00"] > 0:
        dx = 14 - m["m10"] / m["m00"]
        dy = 14 - m["m01"] / m["m00"]
        traslado = np.float32([[1, 0, dx], [0, 1, dy]])
        lienzo = cv2.warpAffine(lienzo, traslado, (28, 28), flags=cv2.INTER_LINEAR,
                                borderValue=0)
    return lienzo.astype(np.float32) / 255.0


def procesar(imagen, varios=True, es_lienzo=False, fondo_transparente=255):
    """Imagen (gris/RGB/RGBA uint8) -> Resultado con los digitos encontrados.

    varios=True separa varios digitos escritos en una fila (izquierda a derecha);
    varios=False toma todo lo que haya como un unico digito.
    es_lienzo=True desactiva el filtro de bordes de papel (en el lienzo un
    digito grande puede tocar los bordes).
    """
    gris = _tamano_de_trabajo(a_gris(imagen, fondo_transparente))
    realce, invertida, contraste = realzar_trazos(gris, corregir_fondo=not es_lienzo)
    if contraste < CONTRASTE_MINIMO:
        vacia = np.zeros_like(gris)
        return Resultado(gris, vacia, vacia, invertida)

    mascara = _mascara_limpia(realce)
    etiquetas, componentes = _componentes_validos(mascara, es_lienzo)
    if not componentes:
        return Resultado(gris, realce, mascara, invertida)

    grupos = _agrupar_por_columnas(componentes) if varios else _un_solo_grupo(componentes)
    # Descarta grupos muy bajitos (guiones, puntos, ruido) frente al mas alto.
    alto_max = max(g["y2"] - g["y1"] for g in grupos)
    grupos = [g for g in grupos if (g["y2"] - g["y1"]) >= 0.3 * alto_max]

    digitos = []
    for g in grupos:
        zona = etiquetas[g["y1"]:g["y2"], g["x1"]:g["x2"]]
        binaria = np.isin(zona, g["ids"]).astype(np.uint8) * 255
        caja = tuple(int(v) for v in (g["x1"], g["y1"], g["x2"] - g["x1"], g["y2"] - g["y1"]))
        digitos.append(Digito(a_formato_mnist(binaria), caja))
    return Resultado(gris, realce, mascara, invertida, digitos)


def preprocesar_basico(imagen):
    """Enfoque basico: gris -> 28x28 -> [0, 1], sin invertir ni centrar. Se usa para comparar."""
    gris = a_gris(imagen)
    pequena = cv2.resize(gris, (28, 28))
    return (pequena.astype(np.float32) / 255.0)[None, ..., None]


def dibujar_cajas(resultado, etiquetas=None):
    """Imagen RGB de trabajo con la caja (y la prediccion) de cada digito."""
    vista = cv2.cvtColor(resultado.gris, cv2.COLOR_GRAY2RGB)
    grosor = max(2, max(vista.shape[:2]) // 250)
    for i, d in enumerate(resultado.digitos):
        x, y, w, h = d.caja
        cv2.rectangle(vista, (x, y), (x + w, y + h), (255, 75, 75), grosor)
        if etiquetas is not None:
            escala = max(0.6, max(vista.shape[:2]) / 600)
            (_, alto_txt), _ = cv2.getTextSize(str(etiquetas[i]), cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
            # Encima de la caja; si no cabe (digito pegado arriba), debajo.
            y_txt = y - 6 if y - 6 >= alto_txt else min(vista.shape[0] - 2, y + h + alto_txt + 6)
            cv2.putText(vista, str(etiquetas[i]), (x, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 75, 75), grosor)
    return vista


def medir_grosor_mnist(imagenes):
    """Mediana del grosor relativo en imagenes MNIST (n, 28, 28) uint8.
    Sirve para calibrar GROSOR_RELATIVO_MNIST."""
    relativos = []
    for img in imagenes:
        binaria = np.where(img > 127, 255, 0).astype(np.uint8)
        if np.count_nonzero(binaria) == 0:
            continue
        canonica = _a_escala_canonica(_recortar(binaria))
        relativos.append(_grosor(canonica) / LADO_CANONICO)
    return float(np.median(relativos))
