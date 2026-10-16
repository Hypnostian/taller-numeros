"""Entrena la red convolucional de digitos (MNIST) que usa la app.

Sigue los pasos del cuaderno 8 A2 (cargar MNIST, normalizar, CNN con dos
capas Conv2D + MaxPooling, Adam, EarlyStopping y guardar en .keras) con tres
cambios para que funcione mejor con digitos dibujados o fotografiados:

  * Aumentacion de datos: rotacion, traslacion y zoom pequenos, y grosor de
    trazo variable (como un pincel o marcador mas grueso).
  * Dropout antes de las capas densas.
  * Validacion separada del set de prueba: las 10.000 imagenes de prueba solo
    se usan al final, asi la precision reportada es honesta.

Uso:
    python entrenar_modelo.py            # hasta 30 epocas con parada anticipada
    python entrenar_modelo.py --epocas 5 # prueba rapida
"""

import argparse
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf

CARPETA_MODELO = Path(__file__).parent / "modelo"
SEMILLA = 42
TAMANO_LOTE = 32
N_VALIDACION = 6000


def cargar_datos():
    """MNIST: 60.000 de entrenamiento (54.000 + 6.000 de validacion) y 10.000 de prueba."""
    (x_ent, y_ent), (x_pru, y_pru) = tf.keras.datasets.mnist.load_data()

    def normalizar(x):
        return (x.astype("float32") / 255.0)[..., None]  # [0, 1] y canal de profundidad

    orden = np.random.default_rng(SEMILLA).permutation(len(x_ent))
    val, ent = orden[:N_VALIDACION], orden[N_VALIDACION:]
    return (
        (normalizar(x_ent[ent]), y_ent[ent]),
        (normalizar(x_ent[val]), y_ent[val]),
        (normalizar(x_pru), y_pru),
        x_pru,  # sin normalizar, para guardar ejemplos de errores
    )


def variar_grosor(x):
    """Engrosa el trazo de parte del lote: la dilatacion en escala de grises es
    un max pooling 3x3 con paso 1. 50 % igual, 30 % grueso, 20 % muy grueso."""
    grueso = tf.nn.max_pool2d(x, 3, 1, "SAME")
    muy_grueso = tf.nn.max_pool2d(grueso, 3, 1, "SAME")
    azar = tf.random.uniform([tf.shape(x)[0], 1, 1, 1])
    return tf.where(azar < 0.5, x, tf.where(azar < 0.8, grueso, muy_grueso))


def crear_aumentacion():
    # fill_mode="constant" rellena con negro, el mismo fondo de MNIST.
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04, fill_mode="constant"),  # +-14 grados
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
        tf.keras.layers.RandomZoom(0.15, fill_mode="constant"),
    ], name="aumentacion")


def crear_modelo():
    """Arquitectura del cuaderno + Dropout. Los nombres conv1/conv2 los usa la
    app para mostrar los mapas de caracteristicas."""
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv1"),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="pool1"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),
        tf.keras.layers.Flatten(name="aplanar"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(64, activation="relu", name="densa64"),
        tf.keras.layers.Dense(32, activation="relu", name="densa32"),
        tf.keras.layers.Dense(10, activation="softmax", name="salida"),
    ], name="cnn_digitos")
    modelo.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return modelo


def guardar_errores(modelo, x_pru, y_pru, x_pru_crudo, ruta, maximo=24):
    """Guarda algunos digitos de prueba mal clasificados para verlos en la app."""
    pred = modelo.predict(x_pru, batch_size=256, verbose=0)
    clases = pred.argmax(axis=1)
    malos = np.flatnonzero(clases != y_pru)
    elegidos = malos[np.argsort(-pred[malos].max(axis=1))][:maximo]  # los mas "seguros"
    np.savez_compressed(
        ruta,
        imagenes=x_pru_crudo[elegidos],
        real=y_pru[elegidos],
        predicho=clases[elegidos],
        confianza=pred[elegidos].max(axis=1),
    )
    return clases


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epocas", type=int, default=30)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEMILLA)
    CARPETA_MODELO.mkdir(exist_ok=True)

    (x_ent, y_ent), (x_val, y_val), (x_pru, y_pru), x_pru_crudo = cargar_datos()
    print(f"Entrenamiento: {len(x_ent)}  Validacion: {len(x_val)}  Prueba: {len(x_pru)}")

    aumentacion = crear_aumentacion()
    datos_ent = (
        tf.data.Dataset.from_tensor_slices((x_ent, y_ent))
        .shuffle(10000, seed=SEMILLA)
        .batch(TAMANO_LOTE)
        .map(lambda x, y: (aumentacion(variar_grosor(x), training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    datos_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(256)

    modelo = crear_modelo()
    modelo.summary()
    print("Entrenando... (unos minutos en CPU)")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-5),
    ]

    inicio = time.time()
    historial = modelo.fit(datos_ent, epochs=args.epocas,
                           validation_data=datos_val, callbacks=callbacks, verbose=2)
    segundos = time.time() - inicio

    perdida, precision = modelo.evaluate(x_pru, y_pru, batch_size=256, verbose=0)
    print(f"\nPrueba (10.000 imagenes nunca vistas): precision={precision:.4f}  perdida={perdida:.4f}")

    # Guardar el modelo en el formato nativo de Keras (como en el cuaderno)
    modelo.save(CARPETA_MODELO / "numeros_cnn.keras")
    # Arquitectura (JSON) y pesos por separado, tambien como en el cuaderno
    (CARPETA_MODELO / "numeros_cnn.json").write_text(modelo.to_json(), encoding="utf-8")
    modelo.save_weights(CARPETA_MODELO / "numeros_cnn.weights.h5")

    clases = guardar_errores(modelo, x_pru, y_pru, x_pru_crudo,
                             CARPETA_MODELO / "errores_prueba.npz")
    matriz = tf.math.confusion_matrix(y_pru, clases, num_classes=10).numpy()

    hist = {k: [float(v) for v in vals] for k, vals in historial.history.items()}
    (CARPETA_MODELO / "historial.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")

    mejor = int(np.argmin(hist["val_loss"]))
    metricas = {
        "precision_prueba": float(precision),
        "perdida_prueba": float(perdida),
        "precision_validacion": hist["val_accuracy"][mejor],
        "mejor_epoca": mejor + 1,
        "epocas_entrenadas": len(hist["loss"]),
        "segundos_entrenamiento": round(segundos, 1),
        "parametros": int(modelo.count_params()),
        "errores_prueba": int((clases != y_pru).sum()),
        "matriz_confusion": matriz.tolist(),
        "versiones": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "",
        },
        "fecha": date.today().isoformat(),
    }
    (CARPETA_MODELO / "metricas.json").write_text(json.dumps(metricas, indent=1), encoding="utf-8")
    print(f"Modelo y metricas guardados en {CARPETA_MODELO}")


if __name__ == "__main__":
    main()
