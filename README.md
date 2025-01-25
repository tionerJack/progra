# Conversor de Temperatura usando TensorFlow

Este proyecto implementa un modelo de red neuronal simple utilizando TensorFlow para convertir temperaturas entre grados Celsius y Fahrenheit.

## Descripción

El proyecto utiliza una red neuronal de una sola capa para aprender la relación entre temperaturas en grados Celsius y Fahrenheit. El modelo es entrenado con un conjunto de datos que abarca desde -40°C hasta 100°C, permitiendo predicciones precisas en este rango de temperaturas.

## Requisitos

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Estructura del Modelo

- Capa de entrada: 1 neurona (temperatura en Celsius)
- Capa densa: 1 neurona con función de activación lineal
- Salida: Temperatura en Fahrenheit

## Conjunto de Datos

El modelo es entrenado con los siguientes datos:
- Temperaturas de entrada (Celsius): [-40, -10, 0, 8, 15, 22, 38, 45, 50, 60, 70, 80, 90, 100]
- Temperaturas de salida (Fahrenheit): [-40, 14, 32, 46, 59, 72, 100, 113, 122, 140, 158, 176, 194, 212]

## Entrenamiento

El modelo es entrenado usando:
- Optimizador: Adam con tasa de aprendizaje de 0.1
- Función de pérdida: Error cuadrático medio (mean squared error)
- Épocas: 1000

## Uso

1. Ejecuta el notebook Jupyter (far.ipynb)
2. El modelo se entrenará automáticamente
3. Utiliza `modelo.predict()` para hacer predicciones con nuevas temperaturas

## Visualización

El proyecto incluye una gráfica que muestra la magnitud de pérdida durante el entrenamiento, permitiendo visualizar cómo el modelo mejora con cada época.
