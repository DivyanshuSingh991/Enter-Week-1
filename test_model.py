import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model("waste_classifier.h5")


image_folder = r"C:\Users\3055au\Desktop\Coding\Python\Week-1\dataset\test"


classes = [ 'glass', 'metal', 'paper', 'plastic']  # adjust if needed


image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Check if folder is empty
if not image_files:
    print("⚠️ No images found in the folder.")
else:
    print(f"🧩 Found {len(image_files)} images. Processing...\n")


for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    # Make prediction
    pred = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(pred)
    confidence_score = pred[0][predicted_index] * 100  # convert to percentage

    # Print results
    print(f"🖼️ Image: {img_name}")
    print(f"   ➤ Predicted Class: {classes[predicted_index]}")
    print(f"   ➤ Confidence: {confidence_score:.2f}%\n")

print("✅ All predictions completed!")
