import re
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# 1) 训练数据（你们换成自己的 Domain）
# -------------------------
INTENTS = {
    "greet": {
        "examples": ["hi", "hello", "hey", "good morning", "good evening"],
        "responses": ["Hi! How can I help you?", "Hello! What can I do for you today?"]
    },
    "bye": {
        "examples": ["bye", "goodbye", "see you", "exit", "quit"],
        "responses": ["Bye! Take care.", "See you next time!"]
    },
    "thanks": {
        "examples": ["thanks", "thank you", "tq", "appreciate it"],
        "responses": ["You're welcome!", "No problem!"]
    },
    # ====== 领域相关（示例：UKM FAQ）======
    "ukm_wifi": {
        "examples": [
            "how to connect ukm wifi",
            "wifi not working",
            "cannot connect to ukm internet",
            "ukm wifi password",
            "eduroam login problem"
        ],
        "responses": [
            "Try connecting to the campus WiFi/eduroam and sign in using your UKM credentials. If it still fails, forget the network and reconnect."
        ]
    },
    "ukm_library": {
        "examples": [
            "library opening hours",
            "when does the library open",
            "ukm library time",
            "is the library open today"
        ],
        "responses": [
            "Library hours can vary by day/semester. If you tell me the day (Mon–Sun), I can suggest what to check (official page / notice / counter)."
        ]
    }
}

FALLBACKS = [
    "Sorry, I’m not sure I understand. Can you rephrase?",
    "I didn’t get that. Could you ask in another way?",
    "I can help with common questions. Try asking about WiFi, library, or say 'help'."
]

# -------------------------
# 2) 预处理
# -------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# 3) 训练一个 intent classifier
# -------------------------
def build_model(intents_dict):
    X, y = [], []
    for intent, item in intents_dict.items():
        for ex in item["examples"]:
            X.append(preprocess(ex))
            y.append(intent)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y)
    return vectorizer, clf

VECTORIZER, CLF = build_model(INTENTS)

# -------------------------
# 4) 规则 + ML 混合回复
# -------------------------
def rule_based_reply(user_text: str):
    t = preprocess(user_text)

    # 规则：非常明确的通用意图
    if t in {"hi", "hello", "hey"}:
        return "greet", random.choice(INTENTS["greet"]["responses"]), 1.0
    if t in {"bye", "goodbye", "exit", "quit"}:
        return "bye", random.choice(INTENTS["bye"]["responses"]), 1.0
    if t in {"thanks", "thank you", "tq"}:
        return "thanks", random.choice(INTENTS["thanks"]["responses"]), 1.0

    return None, None, 0.0

def ml_reply(user_text: str):
    t = preprocess(user_text)
    vec = VECTORIZER.transform([t])
    proba = CLF.predict_proba(vec)[0]
    classes = CLF.classes_
    best_idx = int(proba.argmax())
    best_intent = classes[best_idx]
    confidence = float(proba[best_idx])

    # 置信度门槛：太低就 fallback
    if confidence < 0.35:
        return "fallback", random.choice(FALLBACKS), confidence

    return best_intent, random.choice(INTENTS[best_intent]["responses"]), confidence

def get_response(user_text: str):
    intent, resp, conf = rule_based_reply(user_text)
    if intent:
        return intent, resp, conf
    return ml_reply(user_text)

# -------------------------
# 5) Streamlit UI
# -------------------------
st.set_page_config(page_title="TP2633 Chatbot", page_icon="💬", layout="centered")

st.title("💬 TP2633 Chatbot")
st.caption("Rule-based + ML (TF-IDF + Logistic Regression) demo")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("About")
    st.write("This chatbot demonstrates preprocessing, intent classification, and dialog management.")
    st.write("You can replace intents/examples with your chosen domain dataset.")
    if st.button("Clear chat"):
        st.session_state.history = []

# Chat display
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.history.append(("user", user_input))

    intent, reply, conf = get_response(user_input)
    bot_msg = f"{reply}\n\n_(intent: {intent}, confidence: {conf:.2f})_"

    st.session_state.history.append(("assistant", bot_msg))

    with st.chat_message("assistant"):
        st.write(bot_msg)
