from flask import Flask, render_template, request, jsonify
import os

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

IMG_HEIGHT = 180
IMG_WIDTH = 180
class_names = ("кошка", "собака")
print(os.getcwd())
model = load_model('dog_cat_model.keras')

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
  if request.method == "POST":
    img_file = request.files["image"].read()
    img = Image.open(io.BytesIO(img_file))
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return jsonify({
      "predictions":"Это изображение {} с вероятностью {:.2f} процентов.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
      )
    })
if __name__ == "__main__":
  app.run(debug=True)
