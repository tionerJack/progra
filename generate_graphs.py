import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Configurar el estilo de los gráficos
plt.style.use('bmh')  # Usando un estilo incorporado de matplotlib

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

# Gráfico 1: Pérdida durante el entrenamiento
plt.figure(figsize=(12, 6))
plt.plot(historial.history['loss'], linewidth=2, color='blue')
plt.title('Pérdida del Modelo Durante el Entrenamiento', fontsize=14, pad=20)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Magnitud de Pérdida', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')  # Escala logarítmica para mejor visualización
plt.tight_layout()
plt.savefig('images/loss_graph.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 2: Predicciones vs valores reales
plt.figure(figsize=(12, 6))

# Generar predicciones
temp_c = np.linspace(-50, 120, 100)
predicciones = modelo.predict(temp_c)

# Graficar datos de entrenamiento
plt.scatter(centigrados, fahrenheit, color='red', s=100, label='Datos de entrenamiento', zorder=2)

# Graficar línea de predicciones
plt.plot(temp_c, predicciones, color='blue', linewidth=2, label='Predicciones del modelo', zorder=1)

# Añadir detalles al gráfico
plt.title('Predicciones del Modelo vs Valores Reales', fontsize=14, pad=20)
plt.xlabel('Temperatura (°C)', fontsize=12)
plt.ylabel('Temperatura (°F)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Añadir línea de referencia y = (x * 9/5) + 32
temp_real = temp_c * 9/5 + 32
plt.plot(temp_c, temp_real, 'g--', linewidth=1, label='Conversión real', alpha=0.5)

plt.tight_layout()
plt.savefig('images/predictions_graph.png', dpi=300, bbox_inches='tight')
plt.close()

# Imprimir información sobre el modelo
print("\nInformación del modelo:")
print("MSE final:", historial.history['loss'][-1])
print("Pesos:", capa.get_weights()[0][0][0])
print("Sesgo:", capa.get_weights()[1][0])
