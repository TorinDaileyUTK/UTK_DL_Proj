import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import psutil
from sklearn.preprocessing import StandardScaler


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

# Standardizing 'price' and 'duration'
scaler = StandardScaler()
df[['price', 'duration']] = scaler.fit_transform(df[['price', 'duration']])

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
hidden1 = tf.keras.layers.Dense(units=500, activation='sigmoid')(concatenated)
hidden2 = tf.keras.layers.Dense(units=500, activation='sigmoid')(hidden1)
hidden3 = tf.keras.layers.Dense(units=500, activation='sigmoid')(hidden2)

# Output layer
output = tf.keras.layers.Dense(units=1, activation='linear')(hidden3)

# Define the model
model = tf.keras.Model(inputs=[sku_input, order_input, category_input, other_features], outputs=output)

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))


###### run with time and ram usage code ######

def get_ram_usage():
    return psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB

# Print initial RAM usage
print(f"Initial RAM usage: {get_ram_usage():.2f} GB")

start_time = time.time()
# Train the model
#model.fit(x=[df['sku'], df['order'], df['category'], X_other], y=y, epochs=10) #batch_size=32

history = model.fit(x=[df['sku'], df['order'], df['category'], X_other], y=y, epochs=10) #batch_size=32

end_time = time.time()

total_time = end_time - start_time
print(f"Total training time is: {total_time}")

# Print RAM usage after training
print(f"Final RAM usage: {get_ram_usage():.2f} GB")

# Save the model in the default SavedModel format
#model.export('model1')  # This will save the model to a directory named 'my_model'

#uses 12.18gb ram
# Total training time is: 405.3 seconds

# Final loss is 925.27

model.export('model1_s')  # This will save the model to a directory named 'model1_s'



####### Model Evaluation - Loss #######


# Evaluate the model performance
loss = model.evaluate([df['sku'], df['order'], df['category'], X_other], y)
print(f"Model loss: {loss}")


# Plot training loss
pd.DataFrame(history.history['loss']).plot(figsize=(8,5))
plt.title('Training Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()




####### Model Evaluation - R^2 #######

from sklearn.metrics import r2_score

# Define column names manually (update this based on your actual dataset structure)
column_names = ['sku', 'price', 'order', 'duration', 'category', 'quantity']

# Load the CSV without a header and assign column names
df_test = pd.read_csv('pricing_test.csv', names=column_names, header=None)


# Standardizing 'price' and 'duration'
scaler = StandardScaler()
df_test[['price', 'duration']] = scaler.fit_transform(df_test[['price', 'duration']])


# Convert categorical variables into codes (ensure consistency with training data)
# Encode categories using training set categories
df_test['category'] = pd.Categorical(df_test['category'], categories=df['category'].unique()).codes

# Encode SKUs separately for test set (not linked to training SKUs)
df_test['sku'] = df_test['sku'].astype('category').cat.codes

df_test['order'] = pd.Categorical(df_test['order'], categories=df['order'].astype('category').cat.categories).codes

# Extract test features and target variable
X_test = df_test[['sku', 'price', 'order', 'duration', 'category']]
y_test = df_test['quantity']

# Extract test inputs
X_other_test = X_test.drop(columns=['sku', 'order', 'category']).values  # Non-categorical features
X_test_inputs = [df_test['sku'], df_test['order'], df_test['category'], X_other_test]

# Make predictions
y_pred = model.predict(X_test_inputs)

# Compute R² score
r2 = r2_score(y_test, y_pred)
print(f"R² score: {r2:.4f}")



####### Variable Importance ######

# load model:
model = tf.saved_model.load("model1")

#loss function
history = model.fit(x = X, y = y, batch_size=1, epochs = 10)
import pandas as pd
pd.DataFrame(history.history['loss']).plot(figsize=(8,5))

plt.show()


#feature importance

import numpy as np

# check where the input weights were
for var in model.variables:
    print(var.name, var.shape)


#extract weights from first layer whcih is dense/kernel:0 (17, 1000) this was 4th in the above output therefor varaibles[3]
weights = model.variables[3].numpy() 
 

# sum absolute values of weights from first layer
feature_importance = np.sum(np.abs(weights), axis=1)

feature_groups = {
    'sku': feature_importance[:5],       # First 5 values were sku embedding
    'order': feature_importance[5:10],   # Next 5 values were order embedding
    'category': feature_importance[10:15], # Next 5 values were category embedding
    'price': feature_importance[15],     # only one value 
    'duration': feature_importance[16]   # only one vale
}
# aggreagte importance
aggregated_importance = {
    feature: np.sum(values) if isinstance(values, np.ndarray) else values
    for feature, values in feature_groups.items()
}

# Convert to DataFrame
importance_df = pd.DataFrame({
    'Feature': aggregated_importance.keys(),
    'Importance': aggregated_importance.values()
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

#check results
print(importance_df)

# Plot feature importance
importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title('Feature Importance Based on Weights')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.show()

