import streamlit as st
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

st.set_page_config(page_title="News Category Classifier", layout="centered")

# Load model and embedder
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.keras")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embedder

model, embedder = load_resources()
categories = ["World", "Sports", "Business", "Sci/Tech"]

# Utility Functions
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def predict_news_category(text):
    embedded = embedder.encode([text])
    if embedded.shape[1] != 384:
        return "Unknown", 0.0
    lstm_input = np.array(embedded).reshape(1, 24, 16).astype(np.float32)
    predictions = model.predict(lstm_input)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return categories[predicted_index], confidence

# Title
st.title("🗞️ News Category Classifier")

# Navigation buttons
if "mode" not in st.session_state:
    st.session_state.mode = None

st.markdown("### 🔍 Choose What You Want to Do")
col1, col2, col3 = st.columns(3)
if col1.button("🔎 Predict Headline"):
    st.session_state.mode = "single"
if col2.button("📄 Analyze Article"):
    st.session_state.mode = "paragraph"
if col3.button("🎯 Extract by Category"):
    st.session_state.mode = "filter"

st.markdown("---")

# === MODE 1: Single Headline
if st.session_state.mode == "single":
    user_input = st.text_area("✏️ Enter a news headline:")
    if st.button("Predict Category"):
        if not user_input.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Analyzing..."):
                category, conf = predict_news_category(user_input)
            st.success(f"This is **{category}** news. (Confidence: {conf*100:.2f}%)")

# === MODE 2: Full Paragraph
elif st.session_state.mode == "paragraph":
    user_input = st.text_area("📄 Enter your full article:")
    if st.button("Analyze Sentences"):
        if not user_input.strip():
            st.warning("Please enter text.")
        else:
            sentences = split_into_sentences(user_input)
            st.markdown("### 📊 Results:")
            for i, sent in enumerate(sentences):
                category, _ = predict_news_category(sent)
                st.markdown(f"**{i+1}. [{category}]** {sent}")

# === MODE 3: Category Filter
elif st.session_state.mode == "filter":
    user_input = st.text_area("📄 Enter your full article:")
    selected = st.selectbox("🎯 Choose a category", categories)
    if st.button("Show Only This Category"):
        if not user_input.strip():
            st.warning("Please enter text.")
        else:
            sentences = split_into_sentences(user_input)
            filtered = [f"{i+1}. {s}" for i, s in enumerate(sentences) if predict_news_category(s)[0] == selected]
            if filtered:
                st.markdown(f"### 🔎 Showing **{selected}**-related sentences:")
                for line in filtered:
                    st.markdown(line)
            else:
                st.info(f"No results found for **{selected}**.")

st.markdown("---")
st.caption("Built with ❤️ by Junaid Asif")