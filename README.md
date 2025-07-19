# Product Review Sentiment Analyzer

A lightweight Streamlit web app that scores sentiment (positive / neutral / negative) for customer reviews.

**Powered by:** 🤗 Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`.

## Features
* Paste raw reviews or upload a CSV file
* Instant sentiment scoring with confidence
* Downloadable results as CSV

## Quick Start

```bash
# clone the repo
git clone https://github.com/your-user/sentiment_analyzer.git
cd sentiment_analyzer

# create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt

# optional: set your Hugging Face token (improves download speed, avoids rate‑limits)
cp .env.example .env
echo "HF_TOKEN=your_hf_token_here" >> .env

# run locally
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Environment Variables

| Name      | Purpose                                |
|-----------|----------------------------------------|
| `HF_TOKEN`| (Optional) Hugging Face API token for model downloads |

## Deploying

You can deploy to **Streamlit Community Cloud**, **Render**, **Heroku**, or **Railway**.
Make sure to set the `HF_TOKEN` environment variable in your dashboard if you hit download limits.

## License

MIT © 2025 ZaviQ7