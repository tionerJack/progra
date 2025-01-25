![Header](images/header.png)

# DOCTORADO EN INGENIERÍA APLICADA

## ASIGNATURA: SEMINARIO DE PROGRAMACIÓN

### UNIDAD 1
### ACTIVIDAD INTEGRADORA 1: APLICACIÓN PRÁCTICA REDES NEURONALES--


**Nombre del estudiante:** 
 ISMAEL RUFINO GRAJEDA MARÍN
 RUBEN CRUZ GARCIA MENDEZ 
**Matrícula:**  
**Asesor:** DR. DANIEL GONZÁLEZ SCARPULLI

---


## Objetivo
Aplicar los conceptos relacionados con las Redes Neuronales Artificiales (RNA) como uno de los paradigmas utilizados en el aprendizaje automático o aprendizaje de máquina (Machine Learning).

## Fundamento Teórico

Las redes neuronales artificiales son modelos simples del funcionamiento del sistema nervioso. Como define IBM (2021):

> Las redes neuronales artificiales son modelos simples del funcionamiento del sistema nervioso. Las unidades básicas son las neuronas, que generalmente se organizan en capas. Una red neuronal es un modelo simplificado que emula el modo en que el cerebro humano procesa la información: Funciona por el trabajo simultáneo de un número elevado de unidades de procesamiento interconectadas que parecen versiones abstractas de neuronas.

### Estructura de una Red Neuronal
- **Capa de entrada**: Recibe los datos de entrada
- **Capas ocultas**: Procesan la información
- **Capa de salida**: Produce el resultado final
- **Pesos y sesgos**: Se ajustan durante el entrenamiento

## Actividades Realizadas

### Actividad 1: Red Neuronal Simple
Implementación de una red neuronal con una capa de entrada y una de salida para convertir temperaturas.

#### Componentes:
- Capa de entrada: 1 neurona
- Capa de salida: 1 neurona
- Función de activación: Lineal
- Optimizador: Adam (tasa de aprendizaje: 0.1)

#### Código Implementado:
```python
import tensorflow as tf
import numpy as np

# Datos de entrenamiento
centigrados = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])
```

### Actividad 2: Red Neuronal con Capas Ocultas
Modificación del modelo para incluir dos capas ocultas de 3 neuronas cada una.

#### Componentes:
- Capa de entrada: 1 neurona
- Primera capa oculta: 3 neuronas
- Segunda capa oculta: 3 neuronas
- Capa de salida: 1 neurona

### Actividad 3: Red Neuronal Expandida

#### Arquitectura del Modelo
```
Modelo: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 8)                 16        
                                                                
dense_1 (Dense)             (None, 6)                 54        
                                                                
dense_2 (Dense)             (None, 4)                 28        
                                                                
dense_3 (Dense)             (None, 1)                 5         
=================================================================
Total params: 103 (412 bytes)
Trainable params: 103 (412 bytes)
Non-trainable params: 0 (0 bytes)
_________________________________________________________________
```

#### Componentes del Modelo:
1. **Capa de Entrada + Primera Capa Oculta**:
   - Unidades: 8 neuronas
   - Función de activación: ReLU
   - Parámetros: 16 (8 pesos + 8 sesgos)

2. **Segunda Capa Oculta**:
   - Unidades: 6 neuronas
   - Función de activación: ReLU
   - Parámetros: 54 (48 pesos + 6 sesgos)

3. **Tercera Capa Oculta**:
   - Unidades: 4 neuronas
   - Función de activación: ReLU
   - Parámetros: 28 (24 pesos + 4 sesgos)

4. **Capa de Salida**:
   - Unidades: 1 neurona
   - Función de activación: Lineal
   - Parámetros: 5 (4 pesos + 1 sesgo)

#### Hiperparámetros:
- Optimizador: Adam
- Tasa de aprendizaje: 0.01
- Función de pérdida: Error Cuadrático Medio (MSE)
- Épocas de entrenamiento: 1000

#### Datos de Entrenamiento:
- Temperaturas Celsius: [-40, -10, 0, 8, 15, 22, 38, 45, 50, 60, 70, 80, 90, 100]
- Temperaturas Fahrenheit: [-40, 14, 32, 46, 59, 72, 100, 113, 122, 140, 158, 176, 194, 212]

## Resultados

### Análisis de la Función de Pérdida
![Gráfico de Pérdida](images/loss_graph.png)

El gráfico muestra la evolución de la pérdida (error) durante el entrenamiento:
1. **Fase inicial (0-100 épocas)**: Se observa una rápida disminución del error desde aproximadamente 14000 hasta cerca de 500, indicando un aprendizaje acelerado inicial.
2. **Fase intermedia (100-400 épocas)**: La reducción del error se vuelve más gradual, estabilizándose alrededor de 100.
3. **Fase final (400-1000 épocas)**: El error se mantiene estable en un valor muy bajo, indicando que el modelo ha alcanzado un punto óptimo de convergencia.

### Análisis de Resultados
1. El modelo muestra una convergencia exitosa, con una reducción significativa del error a lo largo del entrenamiento
2. La estabilización de la pérdida después de 400 épocas sugiere que el modelo ha encontrado un mínimo local satisfactorio
3. El comportamiento de la curva de pérdida indica que el modelo ha aprendido efectivamente la relación entre las temperaturas

## Conclusiones

1. Las redes neuronales demostraron ser efectivas para aprender relaciones matemáticas, incluso con arquitecturas simples
2. El uso de TensorFlow y Keras facilita la implementación de modelos de redes neuronales
3. La elección de hiperparámetros (tasa de aprendizaje, número de épocas) es crucial para el éxito del entrenamiento
4. El modelo muestra mejor desempeño dentro del rango de datos de entrenamiento
5. La visualización de la pérdida durante el entrenamiento permite verificar el proceso de aprendizaje y la convergencia del modelo

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
