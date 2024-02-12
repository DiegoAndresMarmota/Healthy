import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

pvc = np.array([8, 9, 10, 11, 12], dtype= float)
tam = np.array([65, 70, 85, 80, 85], dtype = float)

# relation = tf.keras.layers.Dense(units=1, input_shape=[1])
# model = tf.keras.Sequential([relation])

initial_input = tf.keras.layers.Dense(units=3, input_shape=[1])
neural_input_1 = tf.keras.layers.Dense(units=3)
neural_input_2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([initial_input, neural_input_1, neural_input_2])

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Relación dirigida por metas en la Reanimación")
loop = model.fit(pvc, tam, epochs=1000, verbose=False)

# Valor del peso y del sesgo del model
print(initial_input.get_weights())
print(neural_input_1.get_weights())
print(neural_input_2.get_weights())

plt.xlabel("N° de Loops")
plt.ylabel("Proyección de la Reanimación")
plt.plot(loop.history['loss'])

print("Relación finalizada")

print("El grado de PVC necesario para mantener una PAM >65mmHg")
prediction = model.predict([8.0])
print("El pronostico es " + str(prediction) + "tam")