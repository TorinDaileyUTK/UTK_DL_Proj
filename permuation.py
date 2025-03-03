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

# Spruce up the plot
import matplotlib.pyplot as plt

# Define colors from your PowerPoint theme
bar_color = "#4143a5"  # Purple-blue shade for bars
edge_color = "white"   # White edges for contrast
text_color = "white"   # White text for readability

# Set dark background style
plt.style.use("dark_background")

# Sort features by importance (largest to smallest)
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
features, importance_values = zip(*sorted_features)  # Unpack sorted data

# Create figure
plt.figure(figsize=(10, 5), dpi=150)

# Plot horizontal bar chart with sorted values
plt.barh(features, importance_values, color=bar_color, edgecolor=edge_color, linewidth=1.2)

# Add labels and title with white text
plt.xlabel("R² Drop (Feature Importance)", fontsize=14, labelpad=10, color=text_color)
plt.ylabel("Feature", fontsize=14, labelpad=10, color=text_color)
plt.title("Feature Importance via Permutation", fontsize=16, fontweight='bold', pad=15, color=text_color)

# Improve tick labels visibility
plt.xticks(fontsize=12, color=text_color)
plt.yticks(fontsize=12, color=text_color)

# Add subtle vertical gridlines for readability
plt.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, color="gray")

# Invert y-axis so the most important feature is at the top
plt.gca().invert_yaxis()

# Display the plot
plt.show()


#### partial dependency
from sklearn.inspection import partial_dependence
import seaborn as sns

# Define a function to create PDP
def plot_pdp(feature, model, df):
    feature_values = np.linspace(df[feature].min(), df[feature].max(), 20)
    pdp_values = []

    for value in feature_values:
        temp_df = df.copy()
        temp_df[feature] = value  

        X_other_temp = temp_df[['price', 'duration']].values.astype(np.float32)
        y_pred = model.predict([temp_df['sku'], temp_df['order'], temp_df['category'], X_other_temp]).ravel()
        
        pdp_values.append(np.mean(y_pred))

    # Plot PDP
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=feature_values, y=pdp_values, color="#4143a5")
    plt.xlabel(feature.capitalize(), fontsize=12, color="white")
    plt.ylabel("Predicted Quantity", fontsize=12, color="white")
    plt.title(f"Partial Dependence Plot for '{feature}'", fontsize=14, color="white")
    plt.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.5, color="gray")
    plt.style.use("dark_background")
    plt.show()

# Create PDP for 'price'
plot_pdp('price', model, df)
plot_pdp('sku', model, df)
plot_pdp('duration', model, df)
plot_pdp('category', model, df)
plot_pdp('order', model, df)
