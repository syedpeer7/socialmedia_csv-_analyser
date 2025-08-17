# Social Media Interactivity & Behavioral Analyzer

An end‑to‑end, *consent‑based* pipeline to measure a candidate's social interactivity, interests, sentiment, toxicity risk, and compute a workplace‑culture fit score.

## ✨ Features
- Engagement metrics (total, per‑day, actions breakdown)
- Sentiment analysis (VADER, lexicon‑based baseline)
- Toxicity flagging (simple keyword list; plug in a stronger model/API)
- Interest categorization (seed keyword buckets)
- LDA topic modeling for exploratory insights
- Streamlit dashboard + downloadable report scaffold

## 🏗️ Structure
```
social-interactivity-analyzer/
  ├─ app.py
  ├─ requirements.txt
  ├─ data/
  │   └─ sample_interactions.csv
  └─ src/
      └─ pipeline.py
```

## 🚀 Quickstart
```bash
# 1) Create virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the dashboard
streamlit run app.py
```

Open the local URL shown by Streamlit to interact with the dashboard.

## 📦 CSV Schema
Required columns:
- `platform` (twitter, instagram, reddit, youtube, linkedin, etc.)
- `user_id`
- `timestamp` (ISO string; UTC preferred)
- `action` (like, comment, post, share)
- `content_text` (text the user wrote or the content caption/headline)
- `content_tags` (comma‑separated hints)
- `url` (optional source link)

## 🧠 Modeling Notes
- **Sentiment**: VADER baseline. For higher accuracy (incl. code‑mixed Hindi‑English), swap with a multilingual transformer (e.g., `cardiffnlp/twitter-xlm-roberta-base-sentiment`) using Hugging Face Transformers.
- **Toxicity**: Replace `DEFAULT_BADWORDS` with a curated list, or integrate Perspective API or a fine‑tuned toxicity classifier.
- **Interests**: `INTEREST_SEED_KEYWORDS` is a minimal seed set. Expand per your domain. For robust taxonomy, consider BERTopic + label mapping.
- **Scoring**: `workplace_culture_fit` is a transparent, editable formula. Calibrate weights with stakeholder feedback and validation sets.

## 🔒 Ethics & Compliance
- Analyze *only* with explicit, informed consent from the candidate.
- Avoid protected attributes and sensitive inferences (religion, caste, politics).
- Offer explainability: show how scores were computed; allow appeals.
- Log actions and keep data retention minimal; provide data deletion tools.

## 🧪 Programmatic Use
```python
from src.pipeline import run_pipeline
res = run_pipeline("data/sample_interactions.csv", n_topics=6)
print(res["indices"])
```

## 🗺️ Next Steps
- Add OAuth flows to pull data with user consent
- Swap VADER with multilingual transformers
- Add PDF report export with charts
- Add per‑platform normalizers (e.g., X vs Instagram behavior)

---

© 2025 Responsible AI demo. For educational use.
