from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import cv2
import numpy as np
from keras.models import load_model
from keras_facenet import FaceNet
import pickle
import os
import random

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
chatbot_model_path = os.path.join("backend", "legal_bot_model")
tokenizer = AutoTokenizer.from_pretrained(chatbot_model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(chatbot_model_path, local_files_only=True).to(device)

# Load face model
face_model = load_model("backend/face_recognition/face_recognition_model.h5")
with open("backend/face_recognition/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
embedder = FaceNet()

authenticated_user = None
CONFIDENCE_THRESHOLD = 0.30

# Smarter intent detection
def classify_intent(text):
    legal_keywords = ["section", "ipc", "article", "law", "petition", "court", "bail", "fir", "act", "suit", "contract"]
    casual_keywords = ["football", "cricket", "how are you", "joke", "fun", "song", "movie", "chat"]
    lower_text = text.lower()

    if any(word in lower_text for word in legal_keywords):
        return "legal"
    elif any(word in lower_text for word in casual_keywords):
        return "casual"
    return "general"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/face-auth", methods=["POST"])
def face_auth():
    global authenticated_user
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    faces = embedder.extract(img, threshold=0.95)

    if not faces:
        return jsonify({"status": "failed", "user": None})

    embedding = faces[0]["embedding"]
    preds = face_model.predict(np.array([embedding]))[0]
    confidence = np.max(preds)
    pred_label = np.argmax(preds)

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({"status": "failed", "user": None})

    user = label_encoder.inverse_transform([pred_label])[0]
    authenticated_user = user
    return jsonify({"status": "success", "user": user})

@app.route("/chat", methods=["POST"])
def chat():
    global authenticated_user
    if not authenticated_user:
        return jsonify({"error": "Face not authenticated"}), 403

    data = request.json
    user_input = data.get("query", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a valid question."})

    intent = classify_intent(user_input)

    if intent == "casual":
        fallback_responses = [
            "Haha, you're keeping it light today. Any legal trouble I can help with?",
            "Let’s talk law if you need it — I’m always here.",
            "Fun! But do you also have a legal question for me?"
        ]
        return jsonify({"response": random.choice(fallback_responses)})

    # Construct smarter prompt
    prompt = (
        f"You are a highly intelligent Indian legal advisor AI. Always give helpful, lawful, accurate answers.\n"
        f"Client: {user_input}\n"
        f"Lawyer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=120,
        do_sample=True,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        num_beams=4,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean hallucinated phrases
    cleanups = [
        "I'm a language model", "As an AI", "I don't have feelings", "I'm not sure",
        "I cannot provide", "as a machine", "currently a student"
    ]
    for bad in cleanups:
        if bad.lower() in response.lower():
            response = "I'm sorry, let's focus on the legal side. Please rephrase your question."

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
