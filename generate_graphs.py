import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento
centigrados = np.array([-40, -10, 0, 8, 15, 22, 38, 45, 50, 60, 70, 80, 90, 100], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100, 113, 122, 140, 158, 176, 194, 212], dtype=float)

# Crear y entrenar el modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenar el modelo y guardar el historial
historial = modelo.fit(centigrados, fahrenheit, epochs=1000, verbose=False)

# Gráfico de pérdida
plt.figure(figsize=(10, 6))
plt.plot(historial.history['loss'])
plt.title('Pérdida del Modelo Durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Magnitud de Pérdida')
plt.grid(True)
plt.savefig('images/loss_graph.png')
plt.close()

# Gráfico de predicciones vs valores reales
plt.figure(figsize=(10, 6))
temp_c = np.linspace(-50, 120, 100)
predicciones = modelo.predict(temp_c)
plt.plot(centigrados, fahrenheit, 'ro', label='Datos de entrenamiento')
plt.plot(temp_c, predicciones, 'b-', label='Predicciones del modelo')
plt.title('Predicciones del Modelo vs Valores Reales')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Temperatura (°F)')
plt.legend()
plt.grid(True)
plt.savefig('images/predictions_graph.png')
plt.close()
