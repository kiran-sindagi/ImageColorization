from flask import Flask, render_template, request
import numpy as np
import cv2
from PIL import Image
import base64
import io

app = Flask(__name__)

# Paths to model files
PROTOTXT = r"./models/models_colorization_deploy_v2.prototxt"
MODEL = r"./models/colorization_release_v2.caffemodel"
POINTS = r"./models/pts_in_hull.npy"

# Load model once
net = cv2.dnn.readNet(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized

def to_base64(img_array):
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_data = buf.getvalue()
    return base64.b64encode(byte_data).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", error="No file uploaded")

        img = Image.open(file.stream).convert("RGB")
        img_np = np.array(img)
        colorized_img = colorizer(img_np)

        original_b64 = to_base64(img_np)
        colorized_b64 = to_base64(colorized_img)

        return render_template(
            "index.html",
            original_image=original_b64,
            colorized_image=colorized_b64
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
