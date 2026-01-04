# CHATBOT INFORMASI KESEHATAN (API)
# Hybrid: LSTM (Intent) + Rule Topic + TF-IDF Similarity
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# config

MODEL_PATH = "model_chatbot_lstm_final.h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
DATASET_PATH = "dataset_kesehatan_fixed.csv"

MAX_LEN = 20
CONFIDENCE_THRESHOLD = 0.60
SIMILARITY_THRESHOLD = 0.20
FINAL_SCORE_THRESHOLD = 0.20

# LOAD MODEL & TOOLS
print("[INFO] Loading model & tools...")

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

df_kb = pd.read_csv(DATASET_PATH)


# PREPROCESSING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# PREPARE TOPICS

topics = set()
for text in df_kb["text"].str.lower():
    words = text.split()
    if len(words) > 0:
        topics.add(words[-1])

topics = sorted(list(topics))

def detect_topic(text):
    text = text.lower()
    for topic in topics:
        if topic in text:
            return topic
    return None

# TF-IDF SETUP

tfidf = TfidfVectorizer()
tfidf.fit(df_kb["text"].str.lower())


# INTENT PREDICTION

def predict_intent(text):
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(pad, verbose=0)
    confidence = float(np.max(pred))
    intent_index = np.argmax(pred)
    intent = label_encoder.inverse_transform([intent_index])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence

    return intent, confidence


# RESPONSE SELECTION

def get_response(intent, user_text, intent_confidence):
    user_text_clean = user_text.lower()

    candidates = df_kb[df_kb["intent"] == intent]
    if len(candidates) == 0:
        return "Maaf, saya belum memiliki informasi tersebut."

    candidate_texts = candidates["text"].str.lower().tolist()
    candidate_vectors = tfidf.transform(candidate_texts)
    user_vector = tfidf.transform([user_text_clean])

    sims = cosine_similarity(user_vector, candidate_vectors)[0]
    best_idx = sims.argmax()
    best_similarity = sims[best_idx]

    final_score = intent_confidence * best_similarity

    if final_score < FINAL_SCORE_THRESHOLD:
        return "Maaf, pertanyaan tersebut belum dapat saya jawab dengan tepat."

    return candidates.iloc[best_idx]["response"]


# API SETUP
app = FastAPI(
    title="Chatbot Informasi Kesehatan API",
    description="Hybrid LSTM + TF-IDF Chatbot",
    version="1.0"
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    intent: str
    confidence: float
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat_api(req: ChatRequest):
    user_input = req.message

    intent, confidence = predict_intent(user_input)

    if intent == "unknown":
        return {
            "intent": "unknown",
            "confidence": float(confidence),
            "response": "Maaf, saya belum memahami pertanyaan tersebut."
        }

    response = get_response(intent, user_input, confidence)

    return {
        "intent": intent,
        "confidence": float(confidence),
        "response": response
    }
