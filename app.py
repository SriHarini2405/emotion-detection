
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import re
import emoji
from deep_translator import GoogleTranslator
from langdetect import detect

st.set_page_config(page_title="Multilingual Emotion Detection", layout="centered")

# Inject pink background and floating emojis
st.markdown("""
    <style>
    body {
        background-color: #ffe6f0 !important;
    }
    .floating-emoji {
        position: fixed;
        animation: floatUp linear infinite;
        pointer-events: none;
        z-index: 9999;
    }
    @keyframes floatUp {
        0% { transform: translateY(100vh); opacity: 1; }
        100% { transform: translateY(-10vh); opacity: 0; }
    }
    </style>
    <script>
    const emojis = ['😊','💖','🎉','✨','🌸','🥰','🎈','❤️','😄','🌈'];
    function createEmoji() {
        const emoji = document.createElement('div');
        emoji.className = 'floating-emoji';
        emoji.innerText = emojis[Math.floor(Math.random() * emojis.length)];
        emoji.style.left = Math.random() * 100 + 'vw';
        emoji.style.fontSize = (20 + Math.random() * 40) + 'px';
        emoji.style.animationDuration = (5 + Math.random() * 5) + 's';
        document.body.appendChild(emoji);
        emoji.addEventListener('animationend', () => emoji.remove());
    }
    setInterval(createEmoji, 400);
    </script>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center; color: #e91e63;'>💬 Multilingual Emotion Detection</h1>", unsafe_allow_html=True)

# Load model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
model.eval()

label_map = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
emoji_map = {
    'joy': '😄', 'sadness': '😢', 'anger': '😡',
    'fear': '😨', 'love': '❤️', 'surprise': '😲'
}
supportive_messages = {
    "joy": "Keep smiling and enjoy every moment! 🌈✨",
    "sadness": "You're not alone. Things will get better 💖",
    "anger": "Breathe... Let it go. Peace begins with you 🌿",
    "fear": "You’ve got this! Face it with courage 💪",
    "love": "Love is powerful – keep spreading it 💌",
    "surprise": "Wow! Life is full of wonders 🎉"
}

def clean_text(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()

def predict_emotion(text):
    try:
        lang = detect(text)
        if lang != "en":
            text = GoogleTranslator(source=lang, target="en").translate(text)
    except:
        lang = 'en'
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs).item()
    return label_map[label_id], probs[0][label_id].item()

text_input = st.text_area("📝 Enter your message below:", height=100)

if st.button("🎯 Detect Emotion"):
    if text_input.strip():
        emotion, confidence = predict_emotion(text_input)
        emoji_icon = emoji_map[emotion]
        supportive = supportive_messages[emotion]

        st.markdown(f"""
            <div style="background-color:#ffe6f9;padding:20px;border-radius:15px;margin-top:20px;">
                <h3 style="color:#e91e63;">{emoji_icon} Emotion: <b>{emotion.capitalize()}</b></h3>
                <p style="color:#555;">📊 Confidence: <b>{confidence*100:.2f}%</b></p>
                <p style="color:#555;">💌 Supportive Message: <i>{supportive}</i></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text.")
