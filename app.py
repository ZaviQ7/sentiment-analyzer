import os
import io
import pandas as pd
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv

# ---- Custom Page Config ----
st.set_page_config(
    page_title="Earring Review Sentiment Analyzer",
    page_icon=":earring:",  # Change to a jewelry/earring emoji if you like!
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
/* Set wider max width and cleaner fonts */
section.main > div { max-width: 850px !important; }
h1, h2, h3, .stApp { font-family: 'Segoe UI', Arial, sans-serif; }
.stButton button { background-color: #4A90E2 !important; color: white !important; border-radius: 6px; }
.stDownloadButton button { background-color: #72B35D !important; color: white !important; border-radius: 6px; }
.stRadio label, .stTextInput label, .stFileUploader label, .stTextArea label {
    font-weight: 600;
}
.stDataFrame { background: #fafcff; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---- Load Model ----
load_dotenv()
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None,
    )
nlp = load_model()

# ---- App Title ----
st.markdown("<h1 style='font-weight:800; color:#333;'>✨ Product Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("Quickly analyze your earring shop's customer reviews to discover what delights your buyers — and what can be improved.")

with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    - **Paste reviews** (one per line) or **upload a CSV** with a `review` column.
    - Click **Analyze Sentiment** to get a summary table.
    - Download the results as a CSV for records or further analysis.
    """)

# ---- Layout ----
input_col, results_col = st.columns([1,2])

with input_col:
    st.subheader("1. Add Reviews")
    input_choice = st.radio("Choose input method:", ("Text area", "Upload CSV"), horizontal=True)
    reviews = []

    if input_choice == "Text area":
        text_data = st.text_area(
            "Paste your reviews (one per line)", height=200, 
            placeholder="Earrings arrived quickly!\nBeautiful packaging.\nToo heavy for my ears..."
        )
        if text_data:
            reviews = [line.strip() for line in text_data.splitlines() if line.strip()]
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "review" not in df.columns:
                st.error("CSV must contain a column named 'review'.")
            else:
                reviews = df["review"].dropna().tolist()

    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

with results_col:
    st.subheader("2. Results")
    if reviews and analyze_btn:
        with st.spinner("Analyzing..."):
            results = nlp(reviews, batch_size=16)
            labels = [max(r, key=lambda x: x["score"])["label"].capitalize() for r in results]
            scores = [round(max(r, key=lambda x: x["score"])["score"], 3) for r in results]
            df_out = pd.DataFrame({
                "Review": reviews,
                "Sentiment": labels,
                "Confidence": scores,
            })
            st.success(f"Analyzed {len(reviews)} reviews!", icon="✅")
            st.dataframe(
                df_out,
                use_container_width=True,
                hide_index=True
            )

            # Download button in sidebar
            st.sidebar.header("📥 Download")
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                "Download results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
            st.toast("Results ready! You can download your CSV from the sidebar.", icon="📥")

    elif not reviews and analyze_btn:
        st.warning("Please enter at least one review or upload a valid CSV.")

# ---- Footer ----
st.markdown(
    "<hr style='margin:2em 0'>"
    "<small style='color:#999;'>MIT License © 2025 ZaviQ7 • Built with Hugging Face 🤗 and Streamlit</small>",
    unsafe_allow_html=True,
)
