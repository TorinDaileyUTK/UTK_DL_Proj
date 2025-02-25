import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# copy torin
df = pd.read_csv('pricing.csv')


df['sku'] = pd.Categorical(df['sku']).codes
df['order'] = pd.Categorical(df['order']).codes
df['category'] = pd.Categorical(df['category']).codes


X = df[['sku', 'price', 'order', 'duration', 'category']]
y = df['quantity']


sku_input = tf.keras.layers.Input(shape=(1,), name='sku_input')
order_input = tf.keras.layers.Input(shape=(1,), name='order_input')
category_input = tf.keras.layers.Input(shape=(1,), name='category_input')

# Embedding layers for categorical variables
sku_embedding = tf.keras.layers.Embedding(input_dim=df['sku'].nunique() + 1, output_dim=5)(sku_input)
order_embedding = tf.keras.layers.Embedding(input_dim=df['order'].nunique() + 1, output_dim=5)(order_input)
category_embedding = tf.keras.layers.Embedding(input_dim=df['category'].nunique() + 1, output_dim=5)(category_input)

# Flatten embeddings
sku_embedding_flat = tf.keras.layers.Flatten()(sku_embedding)
order_embedding_flat = tf.keras.layers.Flatten()(order_embedding)
category_embedding_flat = tf.keras.layers.Flatten()(category_embedding)

# Extract numerical features
X_other = X[['price', 'duration']].values.astype(np.float32)

# Define numerical input layer
other_features = tf.keras.layers.Input(shape=(X_other.shape[1],), name='other_features')

# Concatenate all features
concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, order_embedding_flat, category_embedding_flat, other_features])

# Build model
hidden1 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden2)
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define and compile model
model = tf.keras.Model(inputs=[sku_input, order_input, category_input, other_features], outputs=output)
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

# Train the model
model.fit(x=[df['sku'], df['order'], df['category'], X_other], y=y, epochs=10, batch_size=32, verbose=1)

#get predictions for all predictors
y_pred_full = model.predict([df['sku'], df['order'], df['category'], X_other]).ravel()
r2_full = r2_score(y, y_pred_full)

#save r2
feature_importance = {}

# Permutation importance loop
for feature in X.columns:
    X_shuffled = X.copy()
    np.random.shuffle(X_shuffled[feature].values)  
    X_other_shuffled = X_shuffled[['price', 'duration']].values.astype(np.float32)  
    
    
    y_pred_shuffled = model.predict([X_shuffled['sku'], X_shuffled['order'], X_shuffled['category'], X_other_shuffled]).ravel()
    r2_shuffled = r2_score(y, y_pred_shuffled)
    
    
    feature_importance[feature] = r2_full - r2_shuffled

# Plot
plt.figure(figsize=(10, 5))
plt.barh(list(feature_importance.keys()), list(feature_importance.values()))
plt.xlabel('R² Drop (Feature Importance)')
plt.ylabel('Feature')
plt.title('Feature Importance via Permutation')
plt.show()
