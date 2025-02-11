import pandas as pd
import tensorflow as tf
import matplotlib.pyplot

inputs = tf.keras.layers.Input(shape = (X.shape[1],)) # comma is a trick to make sure it remains a tuple
hidden1 = tf.keras.layers.Dense(units = 2, activation = 'sigmoid', name = 'hidden1')(inputs) # this is a class call
hidden2 = tf.keras.layers.Dense(units = 2, activation = 'sigmoid', name = 'hidden2')(hidden1)
output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden2)

model = tf.keras.Model(inputs = inputs, outputs = output)

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

model.fit(x = X, y = y, batch_size=1, epochs = 10)

history = model.fit(x = X, y = y, batch_size=1, epochs = 10)
import pandas as pd
pd.DataFrame(history.history['loss']).plot(figsize=(8,5))
plt.show()
