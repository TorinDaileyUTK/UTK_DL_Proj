import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import psutil
from sklearn.metrics import r2_score

###### Data Cleaning ######

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
sku_embedding = tf.keras.layers.Embedding(input_dim=df['sku'].nunique() + 1, output_dim=5)(sku_input)
order_embedding = tf.keras.layers.Embedding(input_dim=df['order'].nunique() + 1, output_dim=5)(order_input)
category_embedding = tf.keras.layers.Embedding(input_dim=df['category'].nunique() + 1, output_dim=5)(category_input)

# Flatten the embeddings
sku_embedding_flat = tf.keras.layers.Flatten()(sku_embedding)
order_embedding_flat = tf.keras.layers.Flatten()(order_embedding)
category_embedding_flat = tf.keras.layers.Flatten()(category_embedding)

# Extract numerical features (excluding categorical columns)
X_other = X.drop(columns=['sku', 'order', 'category']).values

# Concatenate embeddings with other features (price, duration, etc.)
other_features = tf.keras.layers.Input(shape=(X_other.shape[1],), name='other_features') # Excluding categorical columns
concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, order_embedding_flat, category_embedding_flat, other_features])



###### Model Building ######



# Add hidden layers
hidden1 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define the model
model = tf.keras.Model(inputs=[sku_input, order_input, category_input, other_features], outputs=output)

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))


model.fit(x=[df['sku'], df['order'], df['category'], X_other], y=y, epochs=10)

y_pred = model.predict([df['sku'], df['order'], df['category'], X_other])

# Calculate R-squared
r2 = r2_score(y, y_pred)









#### remove sku



import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

# Read the dataset
df = pd.read_csv('pricing.csv')

# Convert 'sku', 'order', and 'category' to categorical variables
df['sku'] = pd.Categorical(df['sku']).codes
df['order'] = pd.Categorical(df['order']).codes
df['category'] = pd.Categorical(df['category']).codes

# Define features and target variable (without 'sku')
X = df[['price', 'order', 'duration', 'category']]
y = df['quantity']

# Define input layers
order_input = tf.keras.layers.Input(shape=(1,), name='order_input')
category_input = tf.keras.layers.Input(shape=(1,), name='category_input')

# Define embedding layers
order_embedding = tf.keras.layers.Embedding(input_dim=df['order'].nunique() + 1, output_dim=5)(order_input)
category_embedding = tf.keras.layers.Embedding(input_dim=df['category'].nunique() + 1, output_dim=5)(category_input)

# Flatten embeddings
order_embedding_flat = tf.keras.layers.Flatten()(order_embedding)
category_embedding_flat = tf.keras.layers.Flatten()(category_embedding)

# Extract numerical features (excluding categorical)
X_other = X.drop(columns=['order', 'category']).values.astype(np.float32)

# Ensure X_other is 2D
X_other = X_other.reshape(-1, X_other.shape[1])

# Define numerical input layer
other_features = tf.keras.layers.Input(shape=(X_other.shape[1],), name='other_features')

# Concatenate embeddings with numerical features
concatenated = tf.keras.layers.Concatenate()([order_embedding_flat, category_embedding_flat, other_features])

###### Model Building ######
hidden1 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define model
model = tf.keras.Model(inputs=[order_input, category_input, other_features], outputs=output)

# Compile model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
model.fit(x=[df['order'], df['category'], X_other], y=y, epochs=10, batch_size=32, verbose=1)
 
# Predict
y_predsku = model.predict([df['order'], df['category'], X_other]).ravel()  # Flatten to 1D

# Calculate R²
r2nosku = r2_score(y, y_predsku)
print(f"R² without 'sku': {r2nosku:.4f}")




##### no price 


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

# Read the dataset
df = pd.read_csv('pricing.csv')

# Convert categorical features to numeric codes
df['sku'] = pd.Categorical(df['sku']).codes
df['order'] = pd.Categorical(df['order']).codes
df['category'] = pd.Categorical(df['category']).codes

# Define features and target variable (Keeping 'sku', Removing 'price')
X = df[['sku', 'order', 'duration', 'category']]
y = df['quantity']

# Define input layers
sku_input = tf.keras.layers.Input(shape=(1,), name='sku_input')
order_input = tf.keras.layers.Input(shape=(1,), name='order_input')
category_input = tf.keras.layers.Input(shape=(1,), name='category_input')

# Define embedding layers for categorical variables
sku_embedding = tf.keras.layers.Embedding(input_dim=df['sku'].nunique() + 1, output_dim=5)(sku_input)
order_embedding = tf.keras.layers.Embedding(input_dim=df['order'].nunique() + 1, output_dim=5)(order_input)
category_embedding = tf.keras.layers.Embedding(input_dim=df['category'].nunique() + 1, output_dim=5)(category_input)

# Flatten embeddings
sku_embedding_flat = tf.keras.layers.Flatten()(sku_embedding)
order_embedding_flat = tf.keras.layers.Flatten()(order_embedding)
category_embedding_flat = tf.keras.layers.Flatten()(category_embedding)

# Extract numerical features (excluding categorical)
X_other = X.drop(columns=['sku', 'order', 'category']).values.astype(np.float32)

# Ensure X_other is 2D
if X_other.shape[1] > 0:
    other_features = tf.keras.layers.Input(shape=(X_other.shape[1],), name='other_features')
    inputs = [sku_input, order_input, category_input, other_features]
    concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, order_embedding_flat, category_embedding_flat, other_features])
else:
    inputs = [sku_input, order_input, category_input]
    concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, order_embedding_flat, category_embedding_flat])

###### Model Building ######

# Add hidden layers
hidden1 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define model
model = tf.keras.Model(inputs=inputs, outputs=output)

# Compile model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

# Train model
model.fit(x=[df['sku'], df['order'], df['category'], X_other], y=y, epochs=10, batch_size=32, verbose=1)

# Predict
y_pred_no_price = model.predict([df['sku'], df['order'], df['category'], X_other]).ravel()

# Calculate R² without 'price'
r2_no_price = r2_score(y, y_pred_no_price)
print(f"R² without 'price': {r2_no_price:.4f}")





### no order

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

# Read the dataset
df = pd.read_csv('pricing.csv')

# Convert categorical variables to numerical codes
df['sku'] = pd.Categorical(df['sku']).codes
df['order'] = pd.Categorical(df['order']).codes
df['category'] = pd.Categorical(df['category']).codes

# Define features and target variable (excluding 'order', adding back 'price')
X = df[['sku', 'price', 'duration', 'category']]
y = df['quantity']

# Define input layers
sku_input = tf.keras.layers.Input(shape=(1,), name='sku_input')
category_input = tf.keras.layers.Input(shape=(1,), name='category_input')

# Define embedding layers
sku_embedding = tf.keras.layers.Embedding(input_dim=df['sku'].nunique() + 1, output_dim=5)(sku_input)
category_embedding = tf.keras.layers.Embedding(input_dim=df['category'].nunique() + 1, output_dim=5)(category_input)

# Flatten embeddings
sku_embedding_flat = tf.keras.layers.Flatten()(sku_embedding)
category_embedding_flat = tf.keras.layers.Flatten()(category_embedding)

# Extract numerical features (excluding categorical)
X_other = X.drop(columns=['sku', 'category']).values.astype(np.float32)

# Ensure X_other is 2D
X_other = X_other.reshape(-1, X_other.shape[1])

# Define numerical input layer
other_features = tf.keras.layers.Input(shape=(X_other.shape[1],), name='other_features')

# Concatenate embeddings with numerical features
concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, category_embedding_flat, other_features])

###### Model Building ######
hidden1 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define model
model = tf.keras.Model(inputs=[sku_input, category_input, other_features], outputs=output)

# Compile model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
model.fit(x=[df['sku'], df['category'], X_other], y=y, epochs=10, batch_size=32, verbose=1)
 
# Predict
y_pred_no_order = model.predict([df['sku'], df['category'], X_other]).ravel()  # Flatten to 1D

# Calculate R²
r2_no_order = r2_score(y, y_pred_no_order)
print(f"R² without 'order': {r2_no_order:.4f}")



#### no duration
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

# Read the dataset
df = pd.read_csv('pricing.csv')

# Convert categorical variables to numerical codes
df['sku'] = pd.Categorical(df['sku']).codes
df['order'] = pd.Categorical(df['order']).codes
df['category'] = pd.Categorical(df['category']).codes

# Define features and target variable (excluding 'duration')
X = df[['sku', 'price', 'order', 'category']]  # No 'duration'
y = df['quantity']

# Define input layers
sku_input = tf.keras.layers.Input(shape=(1,), name='sku_input')
order_input = tf.keras.layers.Input(shape=(1,), name='order_input')
category_input = tf.keras.layers.Input(shape=(1,), name='category_input')

# Define embedding layers
sku_embedding = tf.keras.layers.Embedding(input_dim=df['sku'].nunique() + 1, output_dim=5)(sku_input)
order_embedding = tf.keras.layers.Embedding(input_dim=df['order'].nunique() + 1, output_dim=5)(order_input)
category_embedding = tf.keras.layers.Embedding(input_dim=df['category'].nunique() + 1, output_dim=5)(category_input)

# Flatten embeddings
sku_embedding_flat = tf.keras.layers.Flatten()(sku_embedding)
order_embedding_flat = tf.keras.layers.Flatten()(order_embedding)
category_embedding_flat = tf.keras.layers.Flatten()(category_embedding)

# Extract numerical features (excluding categorical)
X_other = X.drop(columns=['sku', 'order', 'category']).values.astype(np.float32)

# Ensure X_other is 2D
X_other = X_other.reshape(-1, X_other.shape[1])

# Define numerical input layer
other_features = tf.keras.layers.Input(shape=(X_other.shape[1],), name='other_features')

# Concatenate embeddings with numerical features
concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, order_embedding_flat, category_embedding_flat, other_features])

###### Model Building ######
hidden1 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define model
model = tf.keras.Model(inputs=[sku_input, order_input, category_input, other_features], outputs=output)

# Compile model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
model.fit(x=[df['sku'], df['order'], df['category'], X_other], y=y, epochs=10, batch_size=32, verbose=1)

# Predict
y_pred_no_duration = model.predict([df['sku'], df['order'], df['category'], X_other]).ravel()  # Flatten to 1D

# Calculate R²
r2_no_duration = r2_score(y, y_pred_no_duration)
print(f"R² without 'duration': {r2_no_duration:.4f}")





#### no category


import pandas as pd
import tensorflow as tf 
import numpy as np
from sklearn.metrics import r2_score

# Read the dataset
df = pd.read_csv('pricing.csv')

# Convert categorical variables to numerical codes
df['sku'] = pd.Categorical(df['sku']).codes
df['order'] = pd.Categorical(df['order']).codes
df['category'] = pd.Categorical(df['category']).codes

# Define features and target variable (excluding 'category')
X = df[['sku', 'price', 'order', 'duration']]  # No 'category'
y = df['quantity']

# Define input layers
sku_input = tf.keras.layers.Input(shape=(1,), name='sku_input')
order_input = tf.keras.layers.Input(shape=(1,), name='order_input')

# Define embedding layers
sku_embedding = tf.keras.layers.Embedding(input_dim=df['sku'].nunique() + 1, output_dim=5)(sku_input)
order_embedding = tf.keras.layers.Embedding(input_dim=df['order'].nunique() + 1, output_dim=5)(order_input)

# Flatten embeddings
sku_embedding_flat = tf.keras.layers.Flatten()(sku_embedding)
order_embedding_flat = tf.keras.layers.Flatten()(order_embedding)

# Extract numerical features (excluding categorical)
X_other = X.drop(columns=['sku', 'order']).values.astype(np.float32)

# Ensure X_other is 2D
X_other = X_other.reshape(-1, X_other.shape[1])

# Define numerical input layer
other_features = tf.keras.layers.Input(shape=(X_other.shape[1],), name='other_features')

# Concatenate embeddings with numerical features
concatenated = tf.keras.layers.Concatenate()([sku_embedding_flat, order_embedding_flat, other_features])

###### Model Building ######
hidden1 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=1000, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define model
model = tf.keras.Model(inputs=[sku_input, order_input, other_features], outputs=output)

# Compile model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
model.fit(x=[df['sku'], df['order'], X_other], y=y, epochs=10, batch_size=32, verbose=1)

# Predict
y_pred_no_category = model.predict([df['sku'], df['order'], X_other]).ravel()  # Flatten to 1D

# Calculate R²
r2_no_category = r2_score(y, y_pred_no_category)
print(f"R² without 'category': {r2_no_category:.4f}")


feature_importance = {
    'Category': r2 - r2_no_category,
    'Duration': r2 - r2_no_duration,
    'Order': r2 - r2_no_order,
    'Price': r2 - r2_no_price,
    'SKU': r2 - r2nosku
}

# Sort features by importance
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Extract names and values
features, importance_values = zip(*sorted_features)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(features, importance_values, color='skyblue')
plt.xlabel("Features")
plt.ylabel("R² Drop (Importance)")
plt.title("Feature Importance Based on R² Drop")
plt.show()


# Spruced up plot
import matplotlib.pyplot as plt

# Define colors from your PowerPoint theme
bar_color = "#3c6d93"  # Blue shade for bars
edge_color = "white"   # White edges for contrast
text_color = "white"   # White text for readability

# Set dark background
plt.style.use("dark_background")

# Create figure
plt.figure(figsize=(10, 6), dpi=150)

# Plot bar chart with customized colors
plt.bar(features, importance_values, color=bar_color, edgecolor=edge_color, linewidth=1.2)

# Add labels and title with white text
plt.xlabel("Features", fontsize=14, labelpad=10, color=text_color)
plt.ylabel("R² Drop (Importance)", fontsize=14, labelpad=10, color=text_color)
plt.title("Feature Importance Based on R² Drop", fontsize=16, fontweight='bold', pad=15, color=text_color)

# Improve tick labels visibility
plt.xticks(rotation=45, ha="right", fontsize=12, color=text_color)
plt.yticks(fontsize=12, color=text_color)

# Add subtle horizontal gridlines
plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, color="gray")

# Display the plot
plt.show()
