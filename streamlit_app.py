import re
import random
import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================
# 1) Domain: University FAQ intents
#    你可以后面再继续加/改 examples 与 responses
# =========================
INTENTS = {
    # --- Common intents (规则更稳) ---
    "greet": {
        "examples": ["hi", "hello", "hey", "good morning", "good evening"],
        "responses": ["Hi! I’m your campus FAQ assistant. How can I help you?", "Hello! Ask me any university FAQ questions."]
    },
    "bye": {
        "examples": ["bye", "goodbye", "see you", "exit", "quit"],
        "responses": ["Bye! Good luck with your studies.", "See you next time!"]
    },
    "thanks": {
        "examples": ["thanks", "thank you", "tq", "appreciate it"],
        "responses": ["You're welcome!", "No problem!"]
    },
    "help": {
        "examples": ["help", "what can you do", "how to use", "menu"],
        "responses": [
            "I can help with common university FAQs such as timetable, exams, library, WiFi, scholarships, and admin services."
        ]
    },

    # --- Domain-specific intents (用 ML 分类更像“智能”FAQ) ---
    "timetable": {
        "examples": [
            "how to check my timetable",
            "where can I see class schedule",
            "timetable for this semester",
            "how do I view my class timetable",
            "schedule for my courses"
        ],
        "responses": [
            "You can usually check your timetable via the university portal/LMS. Look for 'Timetable' or 'Academic' section."
        ]
    },
    "exam": {
        "examples": [
            "when is my exam",
            "exam timetable",
            "final exam schedule",
            "where to check exam date",
            "exam venue"
        ],
        "responses": [
            "Exam schedule and venue are typically posted on the university portal. Check 'Examination' or official announcements."
        ]
    },
    "library": {
        "examples": [
            "library opening hours",
            "is the library open today",
            "how to borrow books",
            "library membership",
            "return books late fine"
        ],
        "responses": [
            "Library info (hours, borrowing, fines) is provided on the library official page/counter. Tell me what you need: hours / borrow / fine."
        ]
    },
    "wifi": {
        "examples": [
            "wifi not working",
            "how to connect campus wifi",
            "eduroam login problem",
            "internet problem in campus",
            "wifi password"
        ],
        "responses": [
            "Try connecting to the campus WiFi/eduroam and sign in with your student credentials. If it fails, forget the network and reconnect."
        ]
    },
    "scholarship": {
        "examples": [
            "how to apply scholarship",
            "financial aid",
            "ptptn application",
            "any bursary available",
            "scholarship requirements"
        ],
        "responses": [
            "Scholarship/financial aid details are usually under Student Affairs/Finance portal. Check eligibility and required documents there."
        ]
    },
    "contact_admin": {
        "examples": [
            "who to contact for student card",
            "how to replace student id",
            "lost student card",
            "where is admin office",
            "how to update personal info"
        ],
        "responses": [
            "For admin issues (ID card, personal info), contact the student service counter/admin office. If you tell me the issue, I’ll suggest the right unit."
        ]
    }
}

FALLBACKS = [
    "Sorry, I’m not sure I understand. Can you rephrase your question?",
    "I didn’t catch that. Could you ask in another way?",
    "I can help with timetable, exams, library, WiFi, scholarships, and admin services. Try one of these topics."
]


# =========================
# 2) Preprocessing
# =========================
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# 3) Build intent classifier (TF-IDF + Logistic Regression)
# =========================
def build_model(intents_dict):
    X, y = [], []
    for intent, item in intents_dict.items():
        for ex in item["examples"]:
            X.append(preprocess(ex))
            y.append(intent)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(max_iter=1500)
    clf.fit(X_vec, y)
    return vectorizer, clf


VECTORIZER, CLF = build_model(INTENTS)


# =========================
# 4) Dialog Management (Hybrid: rules + ML + fallback)
# =========================
def rule_based(user_text: str):
    t = preprocess(user_text)
    if t in {"hi", "hello", "hey"}:
        return "greet", random.choice(INTENTS["greet"]["responses"]), 1.0
    if t in {"bye", "goodbye", "exit", "quit"}:
        return "bye", random.choice(INTENTS["bye"]["responses"]), 1.0
    if t in {"thanks", "thank you", "tq"}:
        return "thanks", random.choice(INTENTS["thanks"]["responses"]), 1.0
    if t in {"help", "menu", "what can you do"}:
        return "help", random.choice(INTENTS["help"]["responses"]), 1.0
    return None, None, 0.0


def ml_predict(user_text: str):
    t = preprocess(user_text)
    vec = VECTORIZER.transform([t])
    proba = CLF.predict_proba(vec)[0]
    classes = CLF.classes_
    best_idx = int(proba.argmax())
    intent = str(classes[best_idx])
    conf = float(proba[best_idx])
    return intent, conf


def get_response(user_text: str, threshold: float = 0.35):
    # 规则优先
    intent, resp, conf = rule_based(user_text)
    if intent:
        return intent, resp, conf

    # ML 意图分类
    intent, conf = ml_predict(user_text)
    if conf < threshold:
        return "fallback", random.choice(FALLBACKS), conf

    resp = random.choice(INTENTS[intent]["responses"])
    return intent, resp, conf


# =========================
# 5) Evaluation utilities
# =========================
def build_testset():
    # 你可以在这里继续扩充测试集（越多越好写 Evaluation）
    rows = []
    for intent, item in INTENTS.items():
        for ex in item["examples"]:
            rows.append({"text": ex, "true_intent": intent})
    return pd.DataFrame(rows)


def evaluate_model(threshold: float):
    df = build_testset()
    preds = []
    for t in df["text"]:
        pred_intent, conf = ml_predict(t)
        # 注意：Evaluation 主要评估 ML 分类，所以这里不走 rule_based
        if conf < threshold:
            preds.append("fallback")
        else:
            preds.append(pred_intent)

    df["pred_intent"] = preds
    # 计算 accuracy（不含 fallback 的更“公平”版本也给你）
    acc_all = accuracy_score(df["true_intent"], df["pred_intent"])

    df_nofb = df[df["pred_intent"] != "fallback"].copy()
    if len(df_nofb) > 0:
        acc_no_fallback = accuracy_score(df_nofb["true_intent"], df_nofb["pred_intent"])
    else:
        acc_no_fallback = 0.0

    labels = sorted(df["true_intent"].unique().tolist())
    cm = confusion_matrix(df["true_intent"], df["pred_intent"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return df, acc_all, acc_no_fallback, cm_df


# =========================
# 6) Streamlit UI
# =========================
st.set_page_config(page_title="University FAQ Chatbot", page_icon="🎓", layout="centered")

st.title("🎓 University FAQ Chatbot (TP2633)")
st.caption("Hybrid: Rule-based + ML Intent Classification (TF-IDF + Logistic Regression)")

page = st.sidebar.radio("Page", ["Chatbot", "Evaluation", "How to Customize"])

threshold = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)

if "history" not in st.session_state:
    st.session_state.history = []

if page == "Chatbot":
    st.subheader("Chat")
    if st.sidebar.button("Clear chat"):
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Ask a university FAQ question...")
    if user_input:
        st.session_state.history.append(("user", user_input))

        intent, reply, conf = get_response(user_input, threshold=threshold)
        bot_msg = f"{reply}\n\n_(intent: {intent}, confidence: {conf:.2f})_"

        st.session_state.history.append(("assistant", bot_msg))
        with st.chat_message("assistant"):
            st.write(bot_msg)

elif page == "Evaluation":
    st.subheader("Evaluation (Intent Classification)")
    df, acc_all, acc_no_fallback, cm_df = evaluate_model(threshold)

    st.write(f"**Accuracy (all samples, including fallback as predicted label):** {acc_all:.2f}")
    st.write(f"**Accuracy (excluding fallback predictions):** {acc_no_fallback:.2f}")
    st.caption("Tip: Increase examples per intent to improve accuracy and reduce fallback.")

    st.write("### Confusion Matrix (ML predictions)")
    st.dataframe(cm_df)

    st.write("### Test Set Predictions")
    st.dataframe(df)

elif page == "How to Customize":
    st.subheader("How to Customize for Your Domain")
    st.markdown(
        """
1) Edit the `INTENTS` dictionary:
- Add 8–12 examples per domain intent
- Keep common intents: greet/bye/thanks/help

2) Improve robustness:
- Add more variations: synonyms, short forms
- Add more domain intents (e.g., registration, fee payment, hostel)

3) Evaluation:
- Extend the test set by adding more examples
- Use the Evaluation page screenshots in your report
        """
    )
