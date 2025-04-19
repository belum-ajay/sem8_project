from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

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
    
    # Debug logs
    print("Input words:", words)
    print("Valid words in GloVe:", [word for word in words if word in embeddings])
    print("Number of valid vectors:", len(valid_vectors))
    
    return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros((dim,))

# Load model and embeddings
model = load_model("text_to_image_generator_cub_character (1).h5")
glove = load_glove_embeddings()

# Debug: check model input shape
print("Model input shape(s):", [inp.shape for inp in model.inputs])

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

        embedding = get_sentence_embedding(desc, glove)
        print("📐 Embedding shape:", embedding.shape)
        print("🔍 Embedding sample values:", embedding[:5])

        embedding = np.expand_dims(embedding, axis=0)

        np.random.seed(seed)
        noise = np.random.normal(0, 1, (1, 100))  # Adjust if your model uses different noise dimensions

        # Predict
        generated = model.predict([noise, embedding])[0]
        generated = (generated * 255).astype(np.uint8)
        image = Image.fromarray(generated)
        image.save("static/output.png")

        return render_template("index.html", image_path="static/output.png")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
