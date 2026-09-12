"""
============================================================
 Trabajo final - Modulos y Paquetes para Machine Learning con Python
 Estudiante: Estrella
 Caso practico: Bursatil Peru Data - prediccion de tendencia de precios
============================================================

Idea general del script:
Arme esto como una pequeña "clase analista" que va cargando datos,
los deja listos, entrena un par de modelos y al final saca sus propias
conclusiones en consola. Preferi organizarlo como clase (en vez de puro
funciones sueltas) porque se parece mas a como uno estructura un
proyecto real, y de paso me sirvio para repasar POO que vimos en el
curso de ingenieria de software.

Si tienes el CSV real de precios, puedes reemplazar el metodo
cargar_datos() para que lea con pd.read_csv("archivo.csv") en vez de
generar datos de prueba.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from scipy.stats import pearsonr

try:
    from nltk.tokenize import word_tokenize
    TENGO_NLTK = True
except Exception:
    TENGO_NLTK = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENGO_TENSORFLOW = True
except Exception:
    TENGO_TENSORFLOW = False


@dataclass
class ResultadoModelo:
    precision: float
    f1: float
    predicciones: np.ndarray


class AnalistaBursatil:
    """
    Encapsula todo el flujo del caso practico: desde los datos crudos
    hasta las graficas finales. La idea de usarla como clase es poder
    ir guardando cosas (el dataframe procesado, el modelo entrenado)
    como atributos, en vez de andar pasando todo por parametro entre
    funciones sueltas.
    """

    PALABRAS_BUENAS = {"crecimiento", "utilidad", "expansion", "solido", "recuperacion"}
    PALABRAS_MALAS = {"perdida", "caida", "riesgo", "deuda", "recesion"}

    def __init__(self, empresa="Bursatil Peru Data", semilla=7):
        self.empresa = empresa
        self.semilla = semilla
        self.datos = None
        self.modelo = None
        self.resultado = None
        self.red_neuronal = None
        self.precision_red_neuronal = None

    # ------------------------------------------------------------
    # Paso 1: datos
    # ------------------------------------------------------------
    def cargar_datos(self, dias=260):
        gen = np.random.default_rng(self.semilla)
        calendario = pd.bdate_range("2024-02-01", periods=dias)

        # simulo una caminata con un pequeño sesgo positivo, como
        # suele comportarse una accion en un periodo sin crisis
        cambios = gen.normal(0.0004, 0.018, size=dias)
        precio = 120 * np.cumprod(1 + cambios)
        volumen = gen.integers(800, 6000, size=dias)

        self.datos = pd.DataFrame({
            "fecha": calendario,
            "precio": precio,
            "volumen": volumen,
        })
        return self

    def preparar_variables(self):
        df = self.datos.copy()
        df["retorno_diario"] = df["precio"].pct_change()
        df["promedio_7d"] = df["precio"].rolling(7).mean()
        df["volatilidad_7d"] = df["retorno_diario"].rolling(7).std()
        # etiqueta: si al dia siguiente el precio termino mas alto, es 1
        df["sube_manana"] = (df["retorno_diario"].shift(-1) > 0).astype(int)
        df = df.dropna().reset_index(drop=True)
        self.datos = df
        return self

    # ------------------------------------------------------------
    # Paso 2: modelo
    # ------------------------------------------------------------
    def entrenar(self):
        variables = ["retorno_diario", "promedio_7d", "volatilidad_7d", "volumen"]
        X = self.datos[variables]
        y = self.datos["sube_manana"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=False
        )

        self.modelo = GradientBoostingClassifier(random_state=self.semilla)
        self.modelo.fit(X_train, y_train)
        pred = self.modelo.predict(X_test)

        self.resultado = ResultadoModelo(
            precision=accuracy_score(y_test, pred),
            f1=f1_score(y_test, pred),
            predicciones=pred,
        )
        print(f"-> {self.empresa}: precision {self.resultado.precision:.2f}, "
              f"f1 {self.resultado.f1:.2f}")
        return self

    # ------------------------------------------------------------
    # Paso 3: Deep Learning (red neuronal de refuerzo)
    # ------------------------------------------------------------
    def entrenar_red_neuronal(self, epocas=15):
        """
        Modelo de Deep Learning con TensorFlow/Keras, pedido como parte
        del entregable. Toma las mismas variables que el modelo de
        Scikit-learn, pero con una red neuronal simple de 2 capas.
        No reemplaza al modelo de Gradient Boosting, lo complementa:
        sirve para comparar si una red logra capturar algo distinto.
        """
        if not TENGO_TENSORFLOW:
            print("-> TensorFlow no esta instalado en este entorno; se omite este paso.")
            return None

        variables = ["retorno_diario", "promedio_7d", "volatilidad_7d", "volumen"]
        X = self.datos[variables].values.astype("float32")
        y = self.datos["sube_manana"].values.astype("float32")

        # normalizo las variables porque las redes neuronales entrenan
        # mucho mejor cuando todas las entradas estan en escalas parecidas
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        red = models.Sequential([
            layers.Input(shape=(X_norm.shape[1],)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        red.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        red.fit(X_norm, y, epochs=epocas, batch_size=16, verbose=0)

        perdida, precision_red = red.evaluate(X_norm, y, verbose=0)
        print(f"-> Red neuronal (Keras): precision {precision_red:.2f}")

        self.red_neuronal = red
        self.precision_red_neuronal = precision_red
        return red

    # ------------------------------------------------------------
    # Paso 4: texto de reportes (NLP + estadistica)
    # ------------------------------------------------------------
    def revisar_reportes(self, reportes):
        puntajes = []
        for texto in reportes:
            texto = texto.lower()
            palabras = word_tokenize(texto) if TENGO_NLTK else texto.split()
            buenas = sum(p in self.PALABRAS_BUENAS for p in palabras)
            malas = sum(p in self.PALABRAS_MALAS for p in palabras)
            puntajes.append(buenas - malas)

        puntajes = np.array(puntajes)
        retornos = self.datos["retorno_diario"].values[: len(puntajes)]
        corr, p_valor = pearsonr(puntajes, retornos)
        print(f"-> Correlacion tono del reporte vs retorno real: "
              f"{corr:.2f} (p={p_valor:.2f})")
        return puntajes, corr, p_valor

    # ------------------------------------------------------------
    # Paso 5: graficos
    # ------------------------------------------------------------
    def graficar(self, salida="grafico_bursatil_peru_data.png"):
        sns.set_theme(style="darkgrid")
        fig, (ax_precio, ax_corr) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax_precio.plot(self.datos["fecha"], self.datos["precio"], color="#0F6E56", label="Precio")
        ax_precio.plot(self.datos["fecha"], self.datos["promedio_7d"], color="#D85A30",
                        linestyle="--", label="Promedio 7 dias")
        ax_precio.set_title(f"Evolucion de precio - {self.empresa}")
        ax_precio.tick_params(axis="x", rotation=40)
        ax_precio.legend()

        matriz = self.datos[["retorno_diario", "promedio_7d", "volatilidad_7d", "volumen"]].corr()
        sns.heatmap(matriz, annot=True, cmap="crest", ax=ax_corr)
        ax_corr.set_title("Correlacion entre variables")

        fig.tight_layout()
        fig.savefig(salida, dpi=150)
        print(f"-> Grafico guardado en {Path(salida).resolve()}")


def main():
    analista = AnalistaBursatil(empresa="Bursatil Peru Data")
    (analista
        .cargar_datos(dias=260)
        .preparar_variables()
        .entrenar())

    analista.entrenar_red_neuronal()

    reportes_de_prueba = [
        "Se observa un crecimiento solido en las utilidades del trimestre",
        "La empresa enfrenta una caida en ventas y aumento de su deuda",
        "Recuperacion moderada pese al riesgo del entorno actual",
    ]
    analista.revisar_reportes(reportes_de_prueba)
    analista.graficar()

    print("\nListo. Con esto queda cubierto el flujo completo del caso practico.")


if __name__ == "__main__":
    main()
