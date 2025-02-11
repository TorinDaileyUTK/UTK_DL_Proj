import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv('pricing.csv')

inputs = df[['sku', 'price', 'order', 'duration', 'category']]
outputs = df['quantity']

inputs = tf.keras.layers.Input(shape = (df.shape[1],)) # comma is a trick to make sure it remains a tuple
hidden1 = tf.keras.layers.Dense(units = 4, activation = 'sigmoid', name = 'hidden1')(inputs) # this is a class call
hidden2 = tf.keras.layers.Dense(units = 4, activation = 'sigmoid', name = 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units = 4, activation = 'sigmoid', name = 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

model.fit(x = X, y = y, batch_size=1, epochs = 10)

#loss function
history = model.fit(x = X, y = y, batch_size=1, epochs = 10)
import pandas as pd
pd.DataFrame(history.history['loss']).plot(figsize=(8,5))
plt.show()
