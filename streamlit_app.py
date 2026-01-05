import re
import random
import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================
# 1) FTSM Domain: FAQ intents (>=20 examples each)
# =========================
INTENTS = {
    "greet": {
        "examples": [
            "hi", "hello", "hey", "good morning", "good evening",
            "hi there", "hello there", "hey there", "morning", "evening",
            "yo", "hola", "assalamualaikum", "salam", "hai",
            "hi bot", "hello bot", "hey chatbot", "good day", "hye"
        ],
        "responses": [
            "Hi! I’m your FTSM FAQ assistant. How can I help you today?",
            "Hello! Ask me anything about FTSM (timetable, exams, labs, UKMFolio, WiFi)."
        ]
    },
    "bye": {
        "examples": [
            "bye", "goodbye", "see you", "see ya", "exit",
            "quit", "close", "end chat", "i have to go", "talk later",
            "bye bye", "see you later", "see u", "cya", "good night",
            "thanks bye", "ok bye", "im leaving", "sign off", "stop"
        ],
        "responses": [
            "Bye! Good luck with your studies!",
            "See you next time. Take care!"
        ]
    },
    "thanks": {
        "examples": [
            "thanks", "thank you", "tq", "thanks a lot", "thank you so much",
            "appreciate it", "much appreciated", "thx", "ty", "thank u",
            "thanks!", "thank you!", "really thanks", "many thanks", "great thanks",
            "thanks bro", "thanks sis", "thank you very much", "nice thanks", "ok thanks"
        ],
        "responses": [
            "You’re welcome!",
            "No problem—happy to help!"
        ]
    },
    "help": {
        "examples": [
            "help", "what can you do", "how to use", "menu", "faq list",
            "what topics", "what can i ask", "show options", "commands", "guide",
            "i need help", "how does this work", "what is this", "can you help me",
            "list faq", "show faq", "what services", "what do you know", "assist", "support"
        ],
        "responses": [
            "I can help with FTSM FAQs: UKMFolio/LMS, timetable, exams, labs, WiFi/eduroam, fees, student card, and admin services."
        ]
    },

    "ukmfolio_lms": {
        "examples": [
            "how to access ukmfolio", "ukmfolio login", "cannot login ukmfolio", "ukmfolio not loading", "ukmfolio down",
            "where to find course materials ukmfolio", "download notes from ukmfolio", "upload assignment ukmfolio", "submit assignment ukmfolio", "where to submit assignment",
            "ukmfolio quiz not showing", "ukmfolio quiz time", "ukmfolio marks", "grade in ukmfolio", "course not appear in ukmfolio",
            "enrol course ukmfolio", "ukmfolio forum", "ukmfolio assignment upload error", "lms ukmfolio problem", "ukmfolio page blank"
        ],
        "responses": [
            "For UKMFolio/LMS: log in with your UKM credentials, then go to 'My courses'. For submission, open the course → Assignment/Quiz → Submit. If the page fails, try refresh, different browser, or clear cache."
        ]
    },
    "timetable": {
        "examples": [
            "how to check my timetable", "how do i check my timetable", "where can i see my timetable", "check timetable", "class timetable",
            "lecture timetable", "timetable for this semester", "schedule for this semester", "my schedule", "class schedule",
            "when is my next class", "what time is my class today", "show my timetable", "view class schedule", "print my timetable",
            "timetable not updated", "timetable changed", "where to view schedule", "how to see class time", "today class schedule"
        ],
        "responses": [
            "To check your timetable, open the SMP system or Folio in the course. "
        ]
    },

    "exam_schedule": {
        "examples": [
            "exam schedule", "exam timetable", "when is my exam", "final exam schedule", "final exam date",
            "exam venue", "where is my exam", "exam location", "exam hall", "exam slip",
            "how to check exam timetable", "exam timetable for semester", "exam schedule for my course", "paper exam date", "exam start time",
            "exam rules", "exam guideline", "exam regulations", "exam entry requirement", "exam attendance",
        ],
        "responses": [
            "Exam timetable/venue is usually published on SMP/official portal announcements. Check the Examination section for your semester/program."
        ]
    },

    "lab_access": {
        "examples": [
            "ftsm lab opening hours", "computer lab opening hours", "is lab open", "lab open today", "lab access",
            "how to enter lab", "need student card for lab", "lab booking", "book lab", "reserve lab",
            "lab schedule", "which lab for class", "lab location ftsm", "where is lab", "computer lab available",
            "can i use lab after class", "lab rules", "lab regulation", "lab computer problem", "lab pc not working", "lab technician"
        ],
        "responses": [
            "For FTSM labs: access and hours depend on the lab and semester schedule. Usually labs follow weekday working hours. For booking/special access, refer to faculty notice or ask the lab technician."
        ]
    },

    "wifi_eduroam": {
        "examples": [
            "wifi not working", "campus wifi not connecting", "ukm wifi problem", "eduroam setup", "eduroam not working",
            "eduroam login problem", "wifi password", "internet slow", "wifi disconnect", "wifi cannot obtain ip",
            "eduroam username", "eduroam password", "how to connect eduroam", "connect wifi in ftsm", "wifi signal weak",
            "internet not available", "wifi authentication failed", "eduroam not showing", "wifi limited access", "wifi no internet"
        ],
        "responses": [
            "For WiFi/eduroam: select the campus WiFi/eduroam network and sign in using your student credentials. If it fails, forget the network and reconnect. If still failing, contact IT/helpdesk."
        ]
    },

    "fees_payment": {
        "examples": [
            "how to pay fees", "tuition fee payment", "fee payment portal", "pay yuran", "payment method for fees",
            "fee deadline", "late fee", "overdue fee", "how to get receipt", "payment receipt",
            "fees not updated", "payment not reflected", "fee statement", "invoice for fees", "how much tuition fee",
            "ptptn payment", "scholarship covers fee", "financial office contact", "installment payment", "fee payment problem"
        ],
        "responses": [
            "For fees/payment, check the student finance/fee section in the portal for invoice and deadlines. After payment, keep the receipt. If payment isn’t reflected, contact the finance office with proof."
        ]
    },

    "student_card": {
        "examples": [
            "lost student card", "student card missing", "replace student id", "reissue student card", "student id replacement",
            "how to make new student card", "student card renewal", "student card problem", "student card not working", "access card not working",
            "where to get student card", "student service counter", "how to report lost card", "lost matric card", "replace matric card",
            "student card photo", "update student id photo", "student id collection", "collect new card", "student id fee"
        ],
        "responses": [
            "If you lost your student card, report it and apply for replacement at the student service counter/admin. Bring identification and follow the official replacement procedure (may include a replacement fee)."
        ]
    },

    "academic_advisor": {
        "examples": [
            "who is my academic advisor", "meet academic advisor", "how to contact advisor", "ftsm academic advisor", "program advisor",
            "need course advice", "course registration help", "advisor consultation", "advisor email", "advisor office hours",
            "how to change course", "drop course advice", "add course advice", "credit hour advice", "study plan advice",
            "program structure", "graduation requirement", "advisor appointment", "who to ask about course", "academic counselling", "need academic help"
        ],
        "responses": [
            "For academic matters (course planning, add/drop, graduation requirements), use SMP. You can usually find advisor info on faculty pages or via your program coordinator."
        ]
    },

    "internship_fyp": {
        "examples": [
            "fyp registration", "final year project", "how to start fyp", "choose fyp supervisor", "fyp supervisor",
            "internship application", "industrial training", "internship requirements", "internship report", "li placement",
            "fyp proposal", "fyp timeline", "fyp deliverables", "fyp meeting", "how to submit fyp report",
            "internship duration", "internship logbook", "internship evaluation", "fyp presentation", "fyp marking", "li briefing"
        ],
        "responses": [
            "For Internship (Industrial Training) / FYP: requirements and timelines are set by your program/department. Check official briefing notes and announcements."
        ]
    },

    "contact_admin": {
        "examples": [
            "where is admin office", "ftsm admin contact", "who to contact for admin", "student affairs office", "office location",
            "how to update personal details", "change phone number", "update address", "update email", "change name in record",
            "request letter", "support letter", "verification letter", "student confirmation letter", "academic letter request",
            "complaint", "submit request", "counter service hours", "when is counter open", "admin email", "call admin"
        ],
        "responses": [
            "For admin requests (letters, profile update, general inquiries), contact the faculty/student service counter during office hours or use the official email/phone listed by FTSM/UKM."
        ]
    }
}

FALLBACKS = [
    "Sorry, I’m not sure I understand. Can you rephrase your question?",
    "I didn’t catch that. Could you ask in another way?",
    "I can help with UKMFolio/LMS, timetable, exams, labs, WiFi/eduroam, fees, student card, advisor, and admin services."
]


# =========================
# 2) Preprocessing (+ tiny typo tolerance)
# =========================
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # typo tolerance (word-level only; won't break "example"/"exact")
    text = re.sub(r"\bexa\s*m\b", "exam", text)   # "exa m" -> "exam"
    text = re.sub(r"\bexa\b", "exam", text)       # "exa" (standalone) -> "exam"

    return text



# =========================
# 3) Build intent classifier
# =========================
def build_model(intents_dict):
    X, y = [], []
    for intent, item in intents_dict.items():
        for ex in item["examples"]:
            X.append(preprocess(ex))
            y.append(intent)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_vec, y)
    return vectorizer, clf


VECTORIZER, CLF = build_model(INTENTS)


# =========================
# 4) Dialog Management (Hybrid)
# =========================
def rule_based(user_text: str):
    t = preprocess(user_text)

    if t in {"hi", "hello", "hey", "hai", "hye"}:
        return "greet", random.choice(INTENTS["greet"]["responses"]), 1.0
    if t in {"bye", "goodbye", "exit", "quit", "stop"}:
        return "bye", random.choice(INTENTS["bye"]["responses"]), 1.0
    if t in {"thanks", "thank you", "tq", "thx", "ty"}:
        return "thanks", random.choice(INTENTS["thanks"]["responses"]), 1.0
    if t in {"help", "menu", "commands", "faq list", "show options"}:
        return "help", random.choice(INTENTS["help"]["responses"]), 1.0

    # keyword safeguards (more robust)
    if "ukmfolio" in t or "lms" in t:
        return "ukmfolio_lms", random.choice(INTENTS["ukmfolio_lms"]["responses"]), 0.99
    if "timetable" in t or "schedule" in t:
        return "timetable", random.choice(INTENTS["timetable"]["responses"]), 0.99
    if ("exam" in t) or ("final" in t and "exam" in t) or ("paper" in t) or ("test" in t):
        return "exam_schedule", random.choice(INTENTS["exam_schedule"]["responses"]), 0.99
    if "eduroam" in t or "wifi" in t or "internet" in t:
        return "wifi_eduroam", random.choice(INTENTS["wifi_eduroam"]["responses"]), 0.99
    if "fee" in t or "fees" in t or "yuran" in t or "payment" in t or "receipt" in t:
        return "fees_payment", random.choice(INTENTS["fees_payment"]["responses"]), 0.99
    if "student card" in t or "matric card" in t or "id card" in t:
        return "student_card", random.choice(INTENTS["student_card"]["responses"]), 0.99
    if "portal" in t:
        return "contact_admin", random.choice(INTENTS["contact_admin"]["responses"]), 0.90
    if "lab" in t:
        return "lab_access", random.choice(INTENTS["lab_access"]["responses"]), 0.99
    if "advisor" in t or "add drop" in t or "graduation" in t or "study plan" in t:
        return "academic_advisor", random.choice(INTENTS["academic_advisor"]["responses"]), 0.90
    if "fyp" in t or "final year project" in t or "internship" in t or "industrial training" in t or "logbook" in t:
        return "internship_fyp", random.choice(INTENTS["internship_fyp"]["responses"]), 0.90
    if "admin" in t or "counter" in t or "letter" in t:
        return "contact_admin", random.choice(INTENTS["contact_admin"]["responses"]), 0.90

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


def get_response(user_text: str, threshold: float = 0.25):
    intent, resp, conf = rule_based(user_text)
    if intent:
        return intent, resp, conf

    intent, conf = ml_predict(user_text)
    if conf < threshold:
        return "fallback", random.choice(FALLBACKS), conf

    resp = random.choice(INTENTS[intent]["responses"])
    return intent, resp, conf


# =========================
# 5) Evaluation
# =========================
def build_testset():
    rows = []
    for intent, item in INTENTS.items():
        for ex in item["examples"]:
            rows.append({"text": ex, "true_intent": intent})
    return pd.DataFrame(rows)


def evaluate_model(threshold: float):
    df = build_testset()

    preds = []
    confs = []
    for txt in df["text"]:
        pred_intent, conf = ml_predict(txt)
        if conf < threshold:
            preds.append("fallback")
        else:
            preds.append(pred_intent)
        confs.append(conf)

    df["pred_intent"] = preds
    df["confidence"] = confs

    acc_all = accuracy_score(df["true_intent"], df["pred_intent"])
    df_nofb = df[df["pred_intent"] != "fallback"].copy()
    acc_no_fallback = accuracy_score(df_nofb["true_intent"], df_nofb["pred_intent"]) if len(df_nofb) else 0.0

    labels = sorted(df["true_intent"].unique().tolist())
    cm = confusion_matrix(df["true_intent"], df["pred_intent"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return df, acc_all, acc_no_fallback, cm_df


# =========================
# 6) Streamlit UI (Chatbot + Evaluation only)
# =========================
st.set_page_config(page_title="FTSM FAQ Chatbot", page_icon="🎓", layout="centered")

st.title("🎓 FTSM FAQ Chatbot (TP2633)")
st.caption("Hybrid Dialog System: Rule-based + ML Intent Classification (TF-IDF + Logistic Regression)")

page = st.sidebar.radio("Page", ["Chatbot", "Evaluation"])
threshold = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)

if "history" not in st.session_state:
    st.session_state.history = []

if page == "Chatbot":
    st.subheader("Chat")
    if st.sidebar.button("Clear chat"):
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Ask a FTSM/UKM university FAQ question...")
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

    st.write(f"**Accuracy (all samples, fallback as label):** {acc_all:.2f}")
    st.write(f"**Accuracy (excluding fallback predictions):** {acc_no_fallback:.2f}")
    st.caption("Tip: Add more examples per intent to improve accuracy and reduce fallback.")

    st.write("### Confusion Matrix (ML predictions)")
    st.dataframe(cm_df)

    st.write("### Test Set Predictions")
    st.dataframe(df)
