import sys
import tensorflow as tf
import numpy as np

MODEL_PATH = "models/flowers_cnn_best.keras"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["chamomile", "tulip", "rose", "sunflower", "dandelion"]

def load_and_prep(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_one.py path/to/image.jpg")
        sys.exit(1)
    image_path = sys.argv[1]
    model = tf.keras.models.load_model(MODEL_PATH)
    x = load_and_prep(image_path)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    print(f"Predicted: {CLASS_NAMES[idx]} (confidence={preds[idx]:.3f})")
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:12s}: {preds[i]:.3f}")

if __name__ == "__main__":
    main()
