import os, json, pandas as pd
import streamlit as st
import plotly.express as px

from pipeline import preprocess, compute_indices, topic_model, top_words_per_topic

st.set_page_config(page_title="Social Interactivity Analyzer", layout="wide")

st.title("📊 Social Media Interactivity & Behavioral Analyzer")

st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV (platform,user_id,timestamp,action,content_text,content_tags,url)", type=["csv"])

if uploaded is None:
    st.info("Using bundled sample dataset. Upload your CSV to analyze your own data.")
    csv_path = os.path.join("data", "sample_interactions.csv")
    df = pd.read_csv(csv_path)
else:
    df = pd.read_csv(uploaded)

df = preprocess(df)

st.subheader("Raw Data")
st.dataframe(df[["platform","user_id","timestamp","action","content_text","content_tags","url"]], use_container_width=True, height=280)

st.subheader("Key Indices")
indices = compute_indices(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Interactions", indices["engagement"]["total"])
col2.metric("Interactions / Day", indices["engagement"]["per_day"])
col3.metric("Positivity Index", indices["positivity_index"])
col4.metric("Toxicity Rate", indices["toxicity_rate"])

st.subheader("Workplace Culture Fit")
st.progress(min(indices["workplace_culture_fit"]/100.0, 1.0), text=f"{indices['workplace_culture_fit']} / 100")

# Sentiment distribution
sent_counts = indices["sentiment_counts"]
sent_df = pd.DataFrame([{"label":k, "count":v} for k,v in sent_counts.items()])
fig_sent = px.bar(sent_df, x="label", y="count", title="Sentiment Counts")
st.plotly_chart(fig_sent, use_container_width=True)

# Interests
interests = indices["interest_distribution"]
int_df = pd.DataFrame([{"interest":k, "percent":v} for k,v in interests.items()]).sort_values("percent", ascending=False)
fig_int = px.pie(int_df, names="interest", values="percent", title="Interest Distribution")
st.plotly_chart(fig_int, use_container_width=True)

# Actions
act = indices["engagement"]["by_action"]
act_df = pd.DataFrame([{"action":k, "count":v} for k,v in act.items()])
fig_act = px.bar(act_df, x="action", y="count", title="Actions Breakdown")
st.plotly_chart(fig_act, use_container_width=True)

# Topics (optional)
st.subheader("Topics (LDA)")
try:
    lda, cv, td = topic_model(df, n_topics=6)
    topics = top_words_per_topic(lda, cv, n_words=8)
    topic_items = [{"topic": k, "top_words": ", ".join(v)} for k, v in topics.items()]
    st.table(pd.DataFrame(topic_items))
except Exception as e:
    st.warning(f"Topic modeling skipped: {e}")

st.caption("⚠️ Ethics: Analyze only with explicit consent. Avoid sensitive categories. This demo uses lexicon-based sentiment and keyword interests; for production, consider robust multilingual transformers and audited toxicity models.")
