from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Image generation grid parameters (match notebook settings)
TEST_ROWS = 4
TEST_COLS = 7
TEST_MARGIN = 16
GENERATE_SQUARE = 64  # Assuming 64x64 output images

# Load GloVe embeddings
def load_glove_embeddings(path="glove.6B.300d.txt"):
    embeddings = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vec
    return embeddings

def get_sentence_embedding(text, embeddings, dim=300):
    words = text.lower().split()
    valid_vectors = [embeddings[word] for word in words if word in embeddings]
    return np.mean(valid_vectors, axis=0).astype(np.float32) if valid_vectors else np.zeros((dim,), dtype=np.float32)

# Load model and embeddings
model = load_model("text_to_image_generator_cub_character (1).h5")
glove = load_glove_embeddings()

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["description"]

        try:
            desc, seed_str = user_input.rsplit(",", 1)
            seed = int(seed_str.strip())
        except ValueError:
            desc = user_input
            seed = np.random.randint(0, 10000)

        print(f"\n📝 Description: {desc}")
        print(f"🌱 Seed: {seed}")

        # Get GloVe embedding
        embedding = get_sentence_embedding(desc, glove)
        embedding = np.repeat(np.expand_dims(embedding, axis=0), 28, axis=0).astype(np.float32)

        # Generate corresponding noise
        np.random.seed(seed)
        noise = np.random.normal(0, 1, (28, 100)).astype(np.float32)

        # Predict
        generated_images = model.predict([noise, embedding])
        generated_image = generated_images[0]  # Get the first image

        # Postprocess
        generated_image = 0.5 * generated_image + 0.5
        generated_image = (generated_image * 255).astype(np.uint8)

        # Save the image
        if not os.path.exists("static"):
            os.makedirs("static")
        image = Image.fromarray(generated_image)
        image_path = "static/output.png"
        image.save(image_path)

        return render_template("index.html", image_path=image_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
