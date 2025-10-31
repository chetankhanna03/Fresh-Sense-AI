# app.py
import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

# TensorFlow / Keras import
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# -----------------------
# Flask Setup
# -----------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
# -----------------------
# Configuration
# -----------------------
MODEL_PATH = "model_final.h5"   # your model file
INPUT_SIZE = (224, 224)

# These are your trained classes — make sure order matches your Teachable Machine model!
CLASS_NAMES = [
    "peach_fresh", "pomegranate_fresh",
    "strawberry_fresh", "apple_fresh",
    "banana_fresh", "orange_fresh",
    "peach_rotten", "pomegranate_rotten",
    "strawberry_fresh", "apple_rotten",
    "banana_rotten", "orange_rotten"
]

# -----------------------
# Load Model Once
# -----------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("❌ Error loading model:", e)
        model = None
else:
    print(f"❌ Model file not found at {MODEL_PATH}")


# -----------------------
# Helper Functions
# -----------------------
def preprocess_image(img_pil):
    """Resize and normalize PIL image to model input"""
    img = img_pil.convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print("Incoming request size:", request.content_length)

    if model is None:
        return jsonify({"success": False, "error": "Model not loaded on server."})

    try:
        # Get image from file or base64
        if "file" in request.files:
            img = Image.open(request.files["file"].stream)
        elif "image" in request.form:
            image_data = request.form["image"].split(",")[1]
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            return jsonify({"success": False, "error": "No image received"})

        # Preprocess image
        img_array = preprocess_image(img)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        label = CLASS_NAMES[predicted_index]
        parts = label.split("_")
        fruit = parts[0].capitalize() if len(parts) > 0 else "Unknown"
        status = parts[1].capitalize() if len(parts) > 1 else "Unknown"

        # Return result
        return jsonify({
            "success": True,
            "result": {
                "fruit": fruit,
                "status": status,
                "confidence": confidence,
                "scores": prediction[0].tolist()
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "Image too large for server — try a smaller capture or reduce quality."}), 413
    


# -----------------------
# Run Flask
# -----------------------
if __name__ == "__main__":
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

