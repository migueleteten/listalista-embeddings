from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Debe enviar el campo 'text'"}), 400

    text = data['text']
    embedding = model.encode(text).tolist()
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
