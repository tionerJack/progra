# Proyecto de Machine Learning con TensorFlow

Este repositorio contiene una serie de actividades prácticas utilizando TensorFlow para diferentes aplicaciones de machine learning.

## Estructura del Proyecto

```
.
├── actividad1.ipynb - Conversor de Temperatura (Celsius a Fahrenheit)
├── actividad2.ipynb - Segunda actividad de machine learning
├── actividad3.ipynb - Tercera actividad de machine learning
├── images/ - Directorio de imágenes y gráficos
└── .venv/ - Entorno virtual de Python
```

## Actividad 1: Conversor de Temperatura

### Descripción
Implementación de una red neuronal simple que aprende a convertir temperaturas de grados Celsius a Fahrenheit.

### Características
- Utiliza TensorFlow y Keras para la implementación del modelo
- Red neuronal con una sola capa densa
- Entrenamiento supervisado con datos de temperatura

### Componentes del Modelo
- **Capa de entrada**: 1 neurona (temperatura en Celsius)
- **Capa densa**: 1 neurona con función de activación lineal
- **Optimizador**: Adam con tasa de aprendizaje de 0.1
- **Función de pérdida**: Error cuadrático medio

### Visualización
- Gráfica de la magnitud de pérdida durante el entrenamiento
- Monitoreo del progreso del aprendizaje

### Gráficos del Modelo

#### Pérdida Durante el Entrenamiento
![Gráfico de Pérdida](images/loss_graph.png)
Este gráfico muestra cómo la pérdida (error) del modelo disminuye durante el entrenamiento, indicando que el modelo está aprendiendo efectivamente la relación entre Celsius y Fahrenheit.

#### Predicciones vs Valores Reales
![Predicciones vs Reales](images/predictions_graph.png)
Este gráfico compara las predicciones del modelo con los valores reales, mostrando la precisión del modelo en diferentes temperaturas.

## Actividad 3: Resultados y Conclusiones

### Resultados
- La implementación exitosa de una red neuronal para la conversión de temperaturas demuestra la capacidad de las redes neuronales para aprender relaciones matemáticas simples.
- El modelo logró una precisión alta en la conversión de temperaturas dentro del rango de entrenamiento.
- La visualización de la pérdida durante el entrenamiento mostró una convergencia efectiva del modelo.

### Conclusiones
- Las redes neuronales, incluso en su forma más simple, pueden ser efectivas para aprender relaciones lineales.
- El uso de TensorFlow y Keras simplifica significativamente la implementación de modelos de aprendizaje automático.
- La elección adecuada de hiperparámetros (tasa de aprendizaje, épocas) es crucial para el éxito del entrenamiento.

## Requisitos

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Instalación

1. Clonar el repositorio
2. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   ```
3. Activar el entorno virtual:
   ```bash
   source .venv/bin/activate  # En Linux/Mac
   ```
4. Instalar las dependencias:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

## Uso

1. Activar el entorno virtual
2. Abrir los notebooks con Jupyter:
   ```bash
   jupyter notebook
   ```
3. Ejecutar las celdas en orden

## Referencias

### Bibliografía
Del Brío, B. M., & Sanz, A. (2006). *Redes neuronales y sistemas borrosos* (3a ed.). Editorial Ra-Ma.

### Direcciones Electrónicas
Amaya, L. (s.f.). Redes NEURALES - curso- Fundamentos - luisamayateacher. Google Sites. https://sites.google.com/site/luisamayateacher/redes-neurales---curso

Barragán, A. (2020, 1 de septiembre). ¿Qué es Una red neuronal artificial? CEBE Belgica. https://cebebelgica.es/es_ES/blog/10/que-es-una-red-neuronal-artificial.html

Documentos de IBM. (2021). IBM - Estados Unidos. https://www.ibm.com/docs/es/spss-modeler/SaaS?topic=networks-neural-model

RingaTech. (s.f.). Tu primera red neuronal en Python y Tensorflow [Video]. Youtube. https://www.youtube.com/watch?v=iX_on3VxZzk
