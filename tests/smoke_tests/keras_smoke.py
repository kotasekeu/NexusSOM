import keras
import numpy as np

model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mse')
model.fit(np.array([[1.0]]), np.array([[1.0]]), epochs=1, verbose=0)
print("Tréninkový test: OK")