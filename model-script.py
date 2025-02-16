import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv('pricing.csv')

X = df[['sku', 'price', 'order', 'duration', 'category']] 
y = df['quantity']

inputs = tf.keras.layers.Input(shape = (X.shape[1],)) # comma is a trick to make sure it remains a tuple
hidden1 = tf.keras.layers.Dense(units = 2, activation = 'sigmoid', name = 'hidden1')(inputs) # this is a class call
hidden2 = tf.keras.layers.Dense(units = 2, activation = 'sigmoid', name = 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units = 2, activation = 'sigmoid', name = 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

model.fit(x = X, y = y, batch_size=1, epochs = 10) 

#loss function
history = model.fit(x = X, y = y, batch_size=1, epochs = 10)
import pandas as pd
pd.DataFrame(history.history['loss']).plot(figsize=(8,5))

plt.show()


#feature importance I think

import numpy as np

for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Weights: {len(layer.get_weights())}")

# extract weights from hidden layer 1 
weights = model.layers[1].get_weights()[0]  

# sum absolute values of weights for each feature 
feature_importance = np.sum(np.abs(weights), axis=1)


importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})


importance_df = importance_df.sort_values(by='Importance', ascending=False)


print(importance_df)


importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title('Feature Importance based on Weights')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.show()
