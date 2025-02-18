import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('pricing.csv')

# Convert 'sku', 'order', and 'category' to categorical variables
df['sku'] = pd.Categorical(df['sku'])
df['order'] = pd.Categorical(df['order'])
df['category'] = pd.Categorical(df['category'])

# Create integer labels for categorical variables
df['sku'] = df['sku'].cat.codes
df['order'] = df['order'].cat.codes
df['category'] = df['category'].cat.codes

# Define features and target variable
X = df[['sku', 'price', 'order', 'duration', 'category']] 
y = df['quantity']

# Define input layers for each feature
sku_input = tf.keras.layers.Input(shape=(1,), name='sku_input')
order_input = tf.keras.layers.Input(shape=(1,), name='order_input')
category_input = tf.keras.layers.Input(shape=(1,), name='category_input')

# Define embedding layers for SKU and category
sku_embedding = tf.keras.layers.Embedding(input_dim=df['sku'].nunique(), output_dim=5)(sku_input)
order_embedding = tf.keras.layers.Embedding(input_dim=df['order'].nunique(), output_dim=5)(order_input)
category_embedding = tf.keras.layers.Embedding(input_dim=df['category'].nunique(), output_dim=5)(category_input)

# Flatten the embeddings
sku_embedding_flat = tf.keras.layers.Flatten()(sku_embedding)
order_embedding_flat = tf.keras.layers.Flatten()(order_embedding)
category_embedding_flat = tf.keras.layers.Flatten()(category_embedding)

# Concatenate embeddings with other features (price, duration, etc.)
other_features = tf.keras.layers.Input(shape=(X.shape[1] - 3,), name='other_features')  # Excluding categorical columns
concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, order_embedding_flat, category_embedding_flat, other_features])

# Add hidden layers
hidden1 = tf.keras.layers.Dense(units=2, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=2, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=2, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define the model
model = tf.keras.Model(inputs=[sku_input, order_input, category_input, other_features], outputs=output)

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

# Prepare the inputs for training (drop categorical columns from X)
X_other = X.drop(columns=['sku', 'order', 'category'])

# Train the model
model.fit(x=[df['sku'], df['order'], df['category'], X_other], y=y, batch_size=32, epochs=5)
 
# Save the model in the default SavedModel format
model.export('model1')  # This will save the model to a directory named 'my_model'

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
