# Model_Training.py
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =====================
# Paths
# =====================
DATA_DIR = "/home/mirna-nageh/Desktop/NTI_SESSIONS/NTI_SESSION_6/archive/flowers"

# =====================
# Dataset
# =====================
img_size = (128, 128)
batch_size = 32
epochs = 20
learning_rate = 0.001

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,   # 20% validation
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Class names
class_names = train_ds.class_names
num_classes = len(class_names)
print("✅ Classes:", class_names)

# Normalize (rescale to 0-1)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# =====================
# Model
# =====================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")  # Multi-class classification
])

# =====================
# Compile
# =====================
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================
# Training
# =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# =====================
# Save Model
# =====================
model.save("flowers_cnn.h5")
print("✅ Model saved as flowers_cnn.h5")
