from __future__ import annotations
import re, os, math, json, datetime as dt
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Lightweight NLP: NLTK VADER for sentiment (lexicon-based)
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -------------------------
# Helpers & preprocessing
# -------------------------

DEFAULT_BADWORDS = {
    "idiot","stupid","nonsense","moron","dumb","trash","shut up","clueless","hate","loser"
}

INTEREST_SEED_KEYWORDS = {
    "environment": ["tree", "plant", "green", "cleanup", "climate", "environment", "recycle"],
    "technology": ["python", "code", "software", "ai", "open-source", "tech", "programming"],
    "health": ["health", "wellness", "yoga", "diet", "mental"],
    "education": ["mentor", "student", "university", "learn", "course", "research"],
    "socialwork": ["charity", "volunteer", "ngo", "community", "drive", "donate"],
    "entertainment": ["movie", "music", "fan", "celebrity", "show"],
    "politics": ["election", "policy", "government", "minister", "party", "politics"],
    "finance": ["stock", "market", "invest", "budget", "finance"],
}

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["text_clean"] = df["content_text"].fillna("").apply(_normalize_text)
    df["action"] = df["action"].str.lower().str.strip()
    df["platform"] = df["platform"].str.lower().str.strip()
    return df

# -------------------------
# Engagement metrics
# -------------------------

def engagement_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"total":0,"by_action":{},"per_day":0.0,"active_days":0}
    total = len(df)
    by_action = df["action"].value_counts().to_dict()
    # approx per-day activity
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    per_day = float(total) / span_days if span_days > 0 else float(total)
    active_days = df["timestamp"].dt.date.nunique()
    return {
        "total": int(total),
        "by_action": {k:int(v) for k,v in by_action.items()},
        "per_day": round(per_day, 2),
        "active_days": int(active_days),
    }

# -------------------------
# Sentiment & Toxicity
# -------------------------

def _ensure_vader() -> SentimentIntensityAnalyzer:
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()

def sentiment_scores(texts: List[str]) -> List[Dict[str,float]]:
    sia = _ensure_vader()
    return [sia.polarity_scores(t or "") for t in texts]

def classify_sentiment(compound: float, pos_threshold=0.2, neg_threshold=-0.2) -> str:
    if compound >= pos_threshold: return "positive"
    if compound <= neg_threshold: return "negative"
    return "neutral"

def toxicity_flags(texts: List[str], badwords: set[str] = DEFAULT_BADWORDS) -> List[Dict[str,Any]]:
    flags = []
    for t in texts:
        t_norm = _normalize_text(t or "")
        hits = [w for w in badwords if w in t_norm]
        flags.append({
            "toxic": bool(hits),
            "hits": hits
        })
    return flags

# -------------------------
# Topic Modeling & Interests
# -------------------------

def topic_model(df: pd.DataFrame, n_topics=6, max_features=2000, min_df=1) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    texts = df["text_clean"].tolist()
    cv = CountVectorizer(max_features=max_features, stop_words="english", min_df=min_df)
    X = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    topic_distrib = lda.fit_transform(X)
    return lda, cv, topic_distrib

def top_words_per_topic(lda: LatentDirichletAllocation, cv: CountVectorizer, n_words=8) -> Dict[int, List[str]]:
    terms = np.array(cv.get_feature_names_out())
    out = {}
    for idx, comp in enumerate(lda.components_):
        top_idx = np.argsort(comp)[-n_words:][::-1]
        out[idx] = terms[top_idx].tolist()
    return out

def keyword_interest_buckets(texts: List[str]) -> Dict[str, int]:
    counts = {k:0 for k in INTEREST_SEED_KEYWORDS}
    for t in texts:
        for cat, kws in INTEREST_SEED_KEYWORDS.items():
            if any(kw in t for kw in kws):
                counts[cat] += 1
    return counts

# -------------------------
# Scoring Engine
# -------------------------

def compute_indices(df: pd.DataFrame) -> Dict[str, Any]:
    em = engagement_metrics(df)
    sents = sentiment_scores(df["text_clean"].tolist())
    df["_compound"] = [d["compound"] for d in sents]
    df["_sent_label"] = df["_compound"].apply(classify_sentiment)
    tox = toxicity_flags(df["text_clean"].tolist())
    df["_toxic"] = [int(d["toxic"]) for d in tox]

    pos = (df["_sent_label"] == "positive").sum()
    neg = (df["_sent_label"] == "negative").sum()
    neu = (df["_sent_label"] == "neutral").sum()
    total = len(df)

    positivity_index = (pos - neg) / total if total else 0.0
    toxicity_rate = df["_toxic"].mean() if total else 0.0

    # Interest distribution via keyword buckets
    interests = keyword_interest_buckets(df["text_clean"].tolist())
    interest_total = sum(interests.values()) or 1
    interest_pct = {k: round(v*100/interest_total, 2) for k,v in interests.items()}

    # Workplace Culture Fit (toy formula; adjust weights)
    # Scale engagement per_day (0..10), positivity (0..1), and toxicity (0..1)
    per_day_scaled = min(em["per_day"]/5.0, 1.0)  # >=5/day clips to 1
    positivity_scaled = (positivity_index + 1)/2   # map -1..1 -> 0..1
    toxicity_scaled = 1.0 - toxicity_rate          # lower toxicity -> higher score

    fit = (0.35*per_day_scaled + 0.45*positivity_scaled + 0.20*toxicity_scaled) * 100.0
    fit = round(float(fit), 2)

    return {
        "engagement": em,
        "sentiment_counts": {"positive": int(pos), "neutral": int(neu), "negative": int(neg)},
        "positivity_index": round(float(positivity_index), 3),
        "toxicity_rate": round(float(toxicity_rate), 3),
        "interest_distribution": interest_pct,
        "workplace_culture_fit": fit,
    }

# -------------------------
# Full pipeline
# -------------------------

def run_pipeline(csv_path: str, n_topics: int = 6) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    df = preprocess(df)

    # Analysis indices
    indices = compute_indices(df)

    # Topic model (optional for insights)
    lda, cv, topic_distrib = topic_model(df, n_topics=n_topics)
    topics = top_words_per_topic(lda, cv, n_words=8)

    # Add top dominant topic per row (for debugging/insight)
    dom_topic = topic_distrib.argmax(axis=1)
    df["_dominant_topic"] = dom_topic

    return {
        "indices": indices,
        "topics": topics,
        "dominant_topic_counts": df["_dominant_topic"].value_counts().to_dict(),
    }

if __name__ == "__main__":
    # Quick test with sample data
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_interactions.csv")
    csv_path = os.path.abspath(csv_path)
    result = run_pipeline(csv_path)
    print(json.dumps(result, indent=2))
