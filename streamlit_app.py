import re
import random
import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


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
            "I can help with FTSM FAQs: UKMFolio/LMS, timetable, exams, labs, WiFi/eduroam, fees, and admin services."
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
            "To check your timetable, open the student portal/academic system → Timetable/Schedule → select semester and your registered courses. You can screenshot or print it from there."
        ]
    },
    "exam_schedule": {
        "examples": [
            "exam schedule", "exam timetable", "when is my exam", "final exam schedule", "final exam date",
            "exam venue", "where is my exam", "exam location", "exam hall", "exam slip",
            "how to check exam timetable", "exam time table for semester", "exam schedule for my course", "paper exam date", "exam start time",
            "exam rules", "exam guideline", "exam regulations", "exam entry requirement", "exam attendance"
        ],
        "responses": [
            "Exam timetable/venue is usually published on SMP. Please check the Examination/Timetable section in SMP for the latest updates."
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
    "academic_ad
